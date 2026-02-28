"""Response predictive model for prompt routing."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter
from pydantic import TypeAdapter

from src.models.scoring.scoring_model_base import ScoringModelBase
from src.data_models.data_models import TrainingData, InputData, OutputData
from src.data_models.response_predictive_types import (
    PreprocessedTrainingDataWithEmbeddings,
    PromptRoutingOutput,
)
from src.models.embedding_specs.embedding_spec_union import EmbeddingSpec
from src.models.embedding_models.embedding_model_base import EmbeddingModelBase
from src.models.optimizers.adamw_spec import AdamWSpec
from src.preprocessing.response_predictive_preprocessor import ResponsePredictivePreprocessor
from src.preprocessing.simple_scaler import SimpleScaler
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.timer import Timer
from src.utils.torch_utils import state_dict_to_cpu
from src.utils.accuracy import compute_pairwise_accuracy
from src.utils.data_split import ValidationSplit, split_response_predictive_preprocessed_data
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models import model_loading
from src.utils.best_model_tracker import BestModelTracker


_DataLoaderType = DataLoader[tuple[
    torch.Tensor,  # prompt_embedding_a
    torch.Tensor,  # prompt_features_a
    torch.Tensor,  # model_embedding_a
    torch.Tensor,  # response_embedding_a
    torch.Tensor,  # response_features_a
    torch.Tensor,  # prompt_embedding_b
    torch.Tensor,  # prompt_features_b
    torch.Tensor,  # model_embedding_b
    torch.Tensor,  # response_embedding_b
    torch.Tensor,  # response_features_b
    torch.Tensor,  # labels
]]


class ResponsePredictiveModel(ScoringModelBase):
    """
    Response-predictive scoring model with three components:
    1. ResponseEncoder: Maps real response embeddings + features to learned representation
    2. ResponsePredictor: Predicts response representation from (prompt, model_embedding)
    3. ResponseScorer: Scores response representations in context of prompts
    """

    def __init__(
        self,
        # Response representation
        response_repr_dim: int = 128,
        encoder_hidden_dims: list[int] | None = None,
        prediction_loss_weight: float = 1.0,
        predictability_loss_weight: float = 0.2,
        repr_kl_loss_weight: float = 0.01,
        # Architecture
        predictor_hidden_dims: list[int] | None = None,
        scorer_hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        # Mixed representation training
        real_repr_ratio: float = 0.8,
        real_repr_decay_per_epoch: float = 0.04,
        min_real_repr_ratio: float = 0.1,
        # Embedding model (for model embeddings)
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_spec: EmbeddingSpec | None = None,
        load_embedding_model_from: str | None = None,
        embedding_model_epochs: int = 10,
        min_model_comparisons: int = 20,
        # Training
        optimizer_spec: OptimizerSpecification | None = None,
        balance_model_samples: bool = True,
        # Standard params
        run_name: str | None = None,
        print_every: int | None = None,
        save_every: int | None = None,
        checkpoint_name: str | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(run_name)

        self.response_repr_dim = response_repr_dim
        self.encoder_hidden_dims = encoder_hidden_dims if encoder_hidden_dims is not None else [256]
        self.prediction_loss_weight = prediction_loss_weight
        self.predictability_loss_weight = predictability_loss_weight
        self.repr_kl_loss_weight = repr_kl_loss_weight
        self.predictor_hidden_dims = predictor_hidden_dims if predictor_hidden_dims is not None else [512, 256]
        self.scorer_hidden_dims = scorer_hidden_dims if scorer_hidden_dims is not None else [256, 128]
        self.dropout = dropout
        self.real_repr_ratio = real_repr_ratio
        self.real_repr_decay_per_epoch = real_repr_decay_per_epoch
        self.min_real_repr_ratio = min_real_repr_ratio
        self.optimizer_spec = optimizer_spec if optimizer_spec is not None else AdamWSpec(learning_rate=0.001)
        self.balance_model_samples = balance_model_samples
        self.embedding_model_name = embedding_model_name
        self.print_every = print_every
        self.save_every = save_every
        self.checkpoint_name = checkpoint_name
        self.min_model_comparisons = min_model_comparisons
        self.embedding_model_epochs = embedding_model_epochs
        self.seed = seed

        self.embedding_spec = embedding_spec
        self.embedding_model: EmbeddingModelBase | None = None

        self.preprocessor = ResponsePredictivePreprocessor(
            embedding_model_name=embedding_model_name,
            min_model_comparisons=min_model_comparisons,
        )

        self._embedding_model_source: str | None = load_embedding_model_from
        self._prompt_embedding_dim: int | None = None
        self._prompt_features_dim: int | None = None
        self._response_embedding_dim: int | None = None
        self._response_features_dim: int | None = None
        self._network: ResponsePredictiveModel._ResponsePredictiveNetwork | None = None
        self._history_entries: list[TrainingHistoryEntry] = []
        self._optimizer_state: dict[str, Any] | None = None
        self._scheduler_state: dict[str, Any] | None = None
        self._epochs_completed: int = 0
        self._prompt_features_scaler: SimpleScaler | None = None
        self._response_features_scaler: SimpleScaler | None = None
        self._best_model_tracker = BestModelTracker()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.last_timer: Timer | None = None

    @property
    def network(self) -> "_ResponsePredictiveNetwork":
        """Get the neural network (must be initialized first)."""
        if self._network is None:
            raise RuntimeError("Network not initialized. Train or load a model first.")
        return self._network

    @property
    def model_embeddings(self) -> dict[str, np.ndarray]:
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not created. Train or load a model first.")

        return self.embedding_model.model_embeddings

    def _initialize_network(
        self,
        prompt_embedding_dim: int,
        prompt_features_dim: int,
        response_embedding_dim: int,
        response_features_dim: int,
    ) -> None:
        """Initialize the network with dimensions."""
        self._prompt_embedding_dim = prompt_embedding_dim
        self._prompt_features_dim = prompt_features_dim
        self._response_embedding_dim = response_embedding_dim
        self._response_features_dim = response_features_dim
        self._network = self._ResponsePredictiveNetwork(
            prompt_embedding_dim=prompt_embedding_dim,
            prompt_features_dim=prompt_features_dim,
            model_embedding_dim=self.embedding_model.embedding_dim,
            response_embedding_dim=response_embedding_dim,
            response_features_dim=response_features_dim,
            response_repr_dim=self.response_repr_dim,
            encoder_hidden_dims=self.encoder_hidden_dims,
            predictor_hidden_dims=self.predictor_hidden_dims,
            scorer_hidden_dims=self.scorer_hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

    def get_config_for_logging(self) -> dict[str, Any]:
        """Get configuration dictionary for Weights & Biases logging."""
        return {
            "model_type": "response_predictive",
            "response_repr_dim": self.response_repr_dim,
            "encoder_hidden_dims": self.encoder_hidden_dims,
            "prediction_loss_weight": self.prediction_loss_weight,
            "predictability_loss_weight": self.predictability_loss_weight,
            "repr_kl_loss_weight": self.repr_kl_loss_weight,
            "predictor_hidden_dims": self.predictor_hidden_dims,
            "scorer_hidden_dims": self.scorer_hidden_dims,
            "dropout": self.dropout,
            "real_repr_ratio": self.real_repr_ratio,
            "real_repr_decay_per_epoch": self.real_repr_decay_per_epoch,
            "min_real_repr_ratio": self.min_real_repr_ratio,
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "balance_model_samples": self.balance_model_samples,
            "embedding_model_name": self.embedding_model_name,
            "preprocessor_version": self.preprocessor.version,
            "embedding_type": self.embedding_model.embedding_type if self.embedding_model is not None else None,
            "embedding_spec": self.embedding_spec.model_dump() if self.embedding_spec is not None else None,
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
        }

    def train(
        self,
        data: TrainingData,
        validation_split: ValidationSplit | None = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        """Train the model."""
        with Timer("train", verbosity="start+end") as train_timer:
            self.last_timer = train_timer

            with Timer("init_or_load_embedding_model", verbosity="start+end", parent=train_timer):
                if self.embedding_model is None:
                    if self._embedding_model_source is not None:
                        self.embedding_model = model_loading.load_embedding_model_from_model(self._embedding_model_source)
                    elif self.embedding_spec is not None:
                        self.embedding_model = self.embedding_spec.create_model(
                            min_model_comparisons=self.min_model_comparisons,
                            preprocessor_seed=self.seed,
                            print_every=self.print_every,
                        )
                    else:
                        raise RuntimeError("No embedding model available and no way to create one")

            with Timer("train_embedding_model", verbosity="start+end", parent=train_timer):
                if not self.embedding_model.is_initialized:
                    self.embedding_model.train(
                        data,
                        validation_split=validation_split,
                        epochs=self.embedding_model_epochs,
                        batch_size=batch_size,
                    )
                elif self.print_every is not None:
                    print("Embedding model is already trained")

            with Timer("preprocess_data", verbosity="start+end", parent=train_timer):
                preprocessed_data = self.preprocessor.preprocess(data)

            self._prompt_features_scaler = SimpleScaler.from_state_dict(preprocessed_data.prompt_scaler_state)
            self._response_features_scaler = SimpleScaler.from_state_dict(preprocessed_data.response_scaler_state)

            if self._network is None:
                prompt_emb_dim = preprocessed_data.pairs[0].prompt_embedding.shape[0]
                response_emb_dim = preprocessed_data.pairs[0].response_embedding_a.shape[0]
                self._initialize_network(
                    prompt_embedding_dim=prompt_emb_dim,
                    prompt_features_dim=preprocessed_data.prompt_features_dim,
                    response_embedding_dim=response_emb_dim,
                    response_features_dim=preprocessed_data.response_features_dim,
                )

            with Timer("add_model_embeddings", verbosity="start+end", parent=train_timer):
                prepared_data = preprocessed_data.add_model_embeddings(self.model_embeddings)

            with Timer("split_preprocessed_data", verbosity="start+end", parent=train_timer):
                preprocessed_train, preprocessed_val = split_response_predictive_preprocessed_data(
                    prepared_data,
                    val_fraction=validation_split.val_fraction if validation_split is not None else 0,
                    seed=validation_split.seed if validation_split is not None else 42,
                )

            with Timer("prepare_dataloaders", verbosity="start+end", parent=train_timer):
                dataloader = self._prepare_dataloader(preprocessed_train, batch_size, use_balancing=True)
                val_dataloader = self._prepare_dataloader(preprocessed_val, batch_size, use_balancing=False) if preprocessed_val is not None else None

            optimizer, scheduler = self._create_optimizer_and_scheduler()

            # Loss functions
            criterion_ranking = nn.MarginRankingLoss(margin=0.1)
            criterion_prediction = nn.CosineEmbeddingLoss(margin=0.0)

            with Timer("epochs", verbosity="start+end", parent=train_timer) as epochs_timer:
                for epoch in range(self._epochs_completed + 1, self._epochs_completed + epochs + 1):
                    result = self._train_epoch(
                        epoch,
                        dataloader,
                        val_dataloader,
                        optimizer,
                        criterion_ranking,
                        criterion_prediction,
                        epochs_timer,
                    )

                    self._log_epoch_result(result)
                    
                    self._best_model_tracker.record_state(
                        accuracy=result.val_accuracy if result.val_accuracy is not None else result.train_accuracy,
                        state_dict=self.get_state_dict(),
                        epoch=epoch
                    )

                    if scheduler is not None:
                        scheduler.step()

                    self._epochs_completed = epoch
            
            # Revert to best model parameters if available
            if self._best_model_tracker.has_best_state:
                print(f"\nReverting to best model parameters from epoch {self._best_model_tracker.best_epoch} (accuracy={self._best_model_tracker.best_accuracy:.4f})")
                self.load_state_dict(self._best_model_tracker.best_state_dict, instance=self)

            self._optimizer_state = optimizer.state_dict()
            self._scheduler_state = scheduler.state_dict() if scheduler is not None else None

            final_metrics = {
                "best_epoch": self._best_model_tracker.best_epoch,
                "best_accuracy": self._best_model_tracker.best_accuracy,
                "total_epochs": self._epochs_completed,
            }

        self.finish_logger_if_needed(
            final_metrics=final_metrics,
            log_timings_from=self.last_timer,
        )

    def predict(
        self,
        X: InputData,
        batch_size: int = 32,
    ) -> OutputData:
        """
        Predict scores for the given prompts and models.
        
        Args:
            X: Input data with prompts and model_names
            batch_size: Batch size for prediction
            
        Returns:
            PromptRoutingOutput with scores for each model
        """
        if self._network is None:
            raise RuntimeError("Model not trained or loaded yet")

        with Timer("predict", verbosity="start+end") as predict_timer:
            self.last_timer = predict_timer
            with Timer("preprocess_input", verbosity="start+end", parent=predict_timer):
                inference_data = self.preprocessor.preprocess_for_inference(
                    prompts=X.prompts,
                    model_names=X.model_names,
                    prompt_scaler=self._prompt_features_scaler,
                    model_embeddings=self.model_embeddings,
                )

            prompt_embeddings = torch.from_numpy(inference_data.prompt_embeddings).to(self.device)  # [n_prompts, prompt_embedding_dim]
            prompt_features = torch.from_numpy(inference_data.prompt_features).to(self.device)  # [n_prompts, prompt_features_dim]
            model_embeddings = torch.from_numpy(inference_data.model_embeddings).to(self.device)  # [n_models, model_embedding_dim]

            self.network.eval()
            scores_dict: dict[str, np.ndarray] = {}

            with torch.no_grad():
                for model_idx, model_name in enumerate(X.model_names):
                    model_emb = model_embeddings[model_idx]  # [model_embedding_dim]
                    model_scores = []

                    for i in range(0, len(prompt_embeddings), batch_size):
                        batch_embeddings = prompt_embeddings[i:i + batch_size]  # [batch_size, prompt_embedding_dim]
                        batch_features = prompt_features[i:i + batch_size]  # [batch_size, prompt_features_dim]
                        batch_size_actual = len(batch_embeddings)

                        batch_model_embs = model_emb.unsqueeze(0).expand(batch_size_actual, -1)  # [batch_size, model_embedding_dim]

                        # Full forward pass: predict response representation, then score it
                        batch_scores = self.network(
                            batch_embeddings,
                            batch_features,
                            batch_model_embs,
                        )  # [batch_size]
                        model_scores.append(batch_scores)

                    all_scores = torch.cat(model_scores)  # [n_prompts]
                    scores_dict[model_name] = all_scores.cpu().numpy()

            return PromptRoutingOutput(_scores=scores_dict)

    def get_history(self) -> TrainingHistory:
        """Get training history."""
        return TrainingHistory.from_entries(self._history_entries)

    def get_state_dict(self) -> dict[str, Any]:
        """
        Get state dictionary for saving the model.
        
        Returns:
            State dictionary containing all model parameters and configuration
        """
        if self._network is None or self._prompt_features_scaler is None or self._response_features_scaler is None:
            raise RuntimeError("Model not trained or loaded yet")

        if not self.embedding_model.is_initialized:
            raise RuntimeError("Embedding model not initialized")

        return {
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "optimizer_state": self._optimizer_state,
            "scheduler_state": self._scheduler_state,
            "response_repr_dim": self.response_repr_dim,
            "encoder_hidden_dims": self.encoder_hidden_dims,
            "prediction_loss_weight": self.prediction_loss_weight,
            "predictability_loss_weight": self.predictability_loss_weight,
            "repr_kl_loss_weight": self.repr_kl_loss_weight,
            "predictor_hidden_dims": self.predictor_hidden_dims,
            "scorer_hidden_dims": self.scorer_hidden_dims,
            "dropout": self.dropout,
            "real_repr_ratio": self.real_repr_ratio,
            "real_repr_decay_per_epoch": self.real_repr_decay_per_epoch,
            "min_real_repr_ratio": self.min_real_repr_ratio,
            "balance_model_samples": self.balance_model_samples,
            "embedding_model_name": self.embedding_model_name,
            "print_every": self.print_every,
            "save_every": self.save_every,
            "checkpoint_name": self.checkpoint_name,
            "preprocessor_version": self.preprocessor.version,
            "prompt_embedding_dim": self._prompt_embedding_dim,
            "prompt_features_dim": self._prompt_features_dim,
            "response_embedding_dim": self._response_embedding_dim,
            "response_features_dim": self._response_features_dim,
            "network_state_dict": state_dict_to_cpu(self.network.state_dict()),
            "epochs_completed": self._epochs_completed,
            "prompt_features_scaler_state": self._prompt_features_scaler.get_state_dict(),
            "response_features_scaler_state": self._response_features_scaler.get_state_dict(),
            "embedding_type": self.embedding_model.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump() if self.embedding_spec is not None else None,
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "embedding_model_state_dict": self.embedding_model.get_state_dict(),
            "seed": self.seed,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any], instance: "ResponsePredictiveModel | None" = None) -> "ResponsePredictiveModel":
        """
        Load model from state dictionary.
        
        Args:
            state_dict: State dictionary from get_state_dict()
            instance: Optional existing model instance to load into
            
        Returns:
            Loaded model instance
        """
        if instance is not None:
            if not isinstance(instance, cls):
                raise TypeError(f"instance must be of type {cls.__name__}, got {type(instance).__name__}")
            model = instance
        else:
            optimizer_spec = OptimizerSpecification.from_serialized(
                state_dict["optimizer_type"],
                state_dict["optimizer_params"],
            )

            # Parse embedding spec using Pydantic TypeAdapter
            embedding_spec_adapter = TypeAdapter(EmbeddingSpec)
            embedding_spec = embedding_spec_adapter.validate_python(state_dict["embedding_spec"]) \
                if state_dict["embedding_spec"] is not None else None

            model = cls(
                response_repr_dim=state_dict["response_repr_dim"],
                encoder_hidden_dims=state_dict["encoder_hidden_dims"],
                prediction_loss_weight=state_dict["prediction_loss_weight"],
                predictability_loss_weight=state_dict.get("predictability_loss_weight", 0.0),
                repr_kl_loss_weight=state_dict.get("repr_kl_loss_weight", 0.0),
                predictor_hidden_dims=state_dict["predictor_hidden_dims"],
                scorer_hidden_dims=state_dict["scorer_hidden_dims"],
                dropout=state_dict["dropout"],
                real_repr_ratio=state_dict["real_repr_ratio"],
                real_repr_decay_per_epoch=state_dict["real_repr_decay_per_epoch"],
                min_real_repr_ratio=state_dict.get("min_real_repr_ratio", 0.0),
                optimizer_spec=optimizer_spec,
                balance_model_samples=state_dict["balance_model_samples"],
                embedding_model_name=state_dict["embedding_model_name"],
                embedding_spec=embedding_spec,
                min_model_comparisons=state_dict["min_model_comparisons"],
                embedding_model_epochs=state_dict["embedding_model_epochs"],
                print_every=state_dict["print_every"],
                save_every=state_dict["save_every"],
                checkpoint_name=state_dict["checkpoint_name"],
                seed=state_dict["seed"],
            )

        model.embedding_model = EmbeddingModelBase.load_from_state_dict(state_dict["embedding_model_state_dict"])
        model._prompt_features_scaler = SimpleScaler.from_state_dict(state_dict["prompt_features_scaler_state"])
        model._response_features_scaler = SimpleScaler.from_state_dict(state_dict["response_features_scaler_state"])

        if model._network is None:
            model._initialize_network(
                prompt_embedding_dim=state_dict["prompt_embedding_dim"],
                prompt_features_dim=state_dict["prompt_features_dim"],
                response_embedding_dim=state_dict["response_embedding_dim"],
                response_features_dim=state_dict["response_features_dim"],
            )
        model.network.load_state_dict(state_dict["network_state_dict"])
        model.network.to(model.device)

        model._optimizer_state = state_dict["optimizer_state"]
        model._scheduler_state = state_dict["scheduler_state"]
        model._epochs_completed = state_dict["epochs_completed"]

        return model

    def _prepare_dataloader(
        self,
        preprocessed_data: PreprocessedTrainingDataWithEmbeddings,
        batch_size: int,
        use_balancing: bool,
    ) -> _DataLoaderType:
        """Prepare dataloader from preprocessed training data."""
        prompt_embeddings_a_list = []
        prompt_features_a_list = []
        model_embeddings_a_list = []
        response_embeddings_a_list = []
        response_features_a_list = []
        prompt_embeddings_b_list = []
        prompt_features_b_list = []
        model_embeddings_b_list = []
        response_embeddings_b_list = []
        response_features_b_list = []
        model_ids_a_list = []
        model_ids_b_list = []
        labels_list = []

        for pair in preprocessed_data.pairs:
            prompt_embeddings_a_list.append(torch.from_numpy(pair.prompt_embedding))
            prompt_features_a_list.append(torch.from_numpy(pair.prompt_features))
            model_embeddings_a_list.append(torch.from_numpy(pair.model_embedding_a))
            response_embeddings_a_list.append(torch.from_numpy(pair.response_embedding_a))
            response_features_a_list.append(torch.from_numpy(pair.response_features_a))

            prompt_embeddings_b_list.append(torch.from_numpy(pair.prompt_embedding))
            prompt_features_b_list.append(torch.from_numpy(pair.prompt_features))
            model_embeddings_b_list.append(torch.from_numpy(pair.model_embedding_b))
            response_embeddings_b_list.append(torch.from_numpy(pair.response_embedding_b))
            response_features_b_list.append(torch.from_numpy(pair.response_features_b))

            model_ids_a_list.append(pair.model_id_a)
            model_ids_b_list.append(pair.model_id_b)

            # label: 1 if model_a should be ranked higher, -1 if model_b should be ranked higher
            labels_list.append(1.0 if pair.winner_label == 0 else -1.0)

        dataset = TensorDataset(
            torch.stack(prompt_embeddings_a_list),
            torch.stack(prompt_features_a_list),
            torch.stack(model_embeddings_a_list),
            torch.stack(response_embeddings_a_list),
            torch.stack(response_features_a_list),
            torch.stack(prompt_embeddings_b_list),
            torch.stack(prompt_features_b_list),
            torch.stack(model_embeddings_b_list),
            torch.stack(response_embeddings_b_list),
            torch.stack(response_features_b_list),
            torch.tensor(labels_list, dtype=torch.float32),
        )

        # Apply weighted sampling if balancing is enabled
        sampler = None
        shuffle = True
        if self.balance_model_samples and use_balancing:
            sampler = self._create_balanced_sampler(model_ids_a_list, model_ids_b_list)
            shuffle = False  # Sampler is mutually exclusive with shuffle

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
        )

    def _create_optimizer_and_scheduler(self) -> tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler | None]:
        """Create optimizer and scheduler."""
        optimizer = self.optimizer_spec.create_optimizer(self.network)
        if self._optimizer_state is not None:
            optimizer.load_state_dict(self._optimizer_state)
            if self.print_every is not None:
                print("Restored optimizer state from checkpoint")

        scheduler = self.optimizer_spec.create_scheduler(optimizer)
        if scheduler is not None and self._scheduler_state is not None:
            scheduler.load_state_dict(self._scheduler_state)
            if self.print_every is not None:
                print("Restored scheduler state from checkpoint")

        return optimizer, scheduler

    def _create_balanced_sampler(
        self,
        model_ids_a: list[int],
        model_ids_b: list[int],
    ) -> WeightedRandomSampler:
        """Create a weighted sampler to balance model representation in training."""
        # Count how many times each model appears
        model_counts = Counter()
        for model_id_a, model_id_b in zip(model_ids_a, model_ids_b):
            model_counts[model_id_a] += 1
            model_counts[model_id_b] += 1

        # Compute weight for each model (inverse frequency)
        model_weights = {
            model_id: 1.0 / count
            for model_id, count in model_counts.items()
        }

        # For each pair, assign weight based on the rarest model in the pair
        sample_weights = []
        for model_id_a, model_id_b in zip(model_ids_a, model_ids_b):
            weight = max(model_weights[model_id_a], model_weights[model_id_b])
            sample_weights.append(weight)

        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

        return WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(sample_weights_tensor),
            replacement=True,
        )

    def _train_epoch(
        self,
        epoch: int,
        dataloader: _DataLoaderType,
        val_dataloader: _DataLoaderType | None,
        optimizer: optim.Optimizer,
        criterion_ranking: nn.Module,
        criterion_prediction: nn.Module,
        epochs_timer: Timer,
    ) -> "ResponsePredictiveModel.EpochResult":
        """Train for one epoch."""
        with Timer(f"epoch_{epoch}", verbosity="start+end", parent=epochs_timer) as timer:
            self.network.train()

            # Compute current real representation ratio
            current_ratio = max(self.min_real_repr_ratio, self.real_repr_ratio - self.real_repr_decay_per_epoch * (epoch - 1))

            # Create seeded generator for reproducible mixing
            generator = torch.Generator(device=self.device).manual_seed(self.seed + epoch)

            # Accumulators for metrics
            total_loss = 0.0
            total_scoring_loss = 0.0
            total_prediction_loss = 0.0
            total_predictability_loss = 0.0
            total_repr_kl_loss = 0.0
            total_train_accuracy = 0.0
            total_prediction_quality = 0.0
            total_scorer_real_repr_accuracy = 0.0
            total_repr_variance = 0.0
            n_batches = 0

            for (
                batch_prompt_emb_a,
                batch_prompt_feat_a,
                batch_model_emb_a,
                batch_response_emb_a,
                batch_response_feat_a,
                batch_prompt_emb_b,
                batch_prompt_feat_b,
                batch_model_emb_b,
                batch_response_emb_b,
                batch_response_feat_b,
                batch_labels,
            ) in dataloader:
                # Move to device
                batch_prompt_emb_a: torch.Tensor = batch_prompt_emb_a.to(self.device)  # [batch, prompt_embedding_dim]
                batch_prompt_feat_a: torch.Tensor = batch_prompt_feat_a.to(self.device)  # [batch, prompt_features_dim]
                batch_model_emb_a: torch.Tensor = batch_model_emb_a.to(self.device)  # [batch, model_embedding_dim]
                batch_response_emb_a: torch.Tensor = batch_response_emb_a.to(self.device)  # [batch, response_embedding_dim]
                batch_response_feat_a: torch.Tensor = batch_response_feat_a.to(self.device)  # [batch, response_features_dim]

                batch_prompt_emb_b: torch.Tensor = batch_prompt_emb_b.to(self.device)  # [batch, prompt_embedding_dim]
                batch_prompt_feat_b: torch.Tensor = batch_prompt_feat_b.to(self.device)  # [batch, prompt_features_dim]
                batch_model_emb_b: torch.Tensor = batch_model_emb_b.to(self.device)  # [batch, model_embedding_dim]
                batch_response_emb_b: torch.Tensor = batch_response_emb_b.to(self.device)  # [batch, response_embedding_dim]
                batch_response_feat_b: torch.Tensor = batch_response_feat_b.to(self.device)  # [batch, response_features_dim]

                batch_labels: torch.Tensor = batch_labels.to(self.device)  # [batch]
                batch_size = len(batch_labels)

                optimizer.zero_grad()

                # 1. Predict response representations
                pred_repr_a = self.network.forward_predict(
                    batch_prompt_emb_a,
                    batch_prompt_feat_a,
                    batch_model_emb_a,
                )  # [batch, response_repr_dim]
                pred_repr_b = self.network.forward_predict(
                    batch_prompt_emb_b,
                    batch_prompt_feat_b,
                    batch_model_emb_b,
                )  # [batch, response_repr_dim]

                # 2. Encode real responses
                real_repr_a = self.network.forward_encode_response(
                    batch_response_emb_a,
                    batch_response_feat_a,
                )  # [batch, response_repr_dim]
                real_repr_b = self.network.forward_encode_response(
                    batch_response_emb_b,
                    batch_response_feat_b,
                )  # [batch, response_repr_dim]

                # 3. Prediction loss: train predictor to match encoder (encoder treated as fixed target)
                target_ones = torch.ones(batch_size, device=self.device)  # [batch]
                pred_loss_a: torch.Tensor = criterion_prediction(pred_repr_a, real_repr_a.detach(), target_ones)
                pred_loss_b: torch.Tensor = criterion_prediction(pred_repr_b, real_repr_b.detach(), target_ones)
                pred_loss = pred_loss_a + pred_loss_b

                # 4. Predictability loss: encourage encoder to produce predictable representations (predictor treated as fixed reference)
                # When predictability_loss_weight=0, encoder gets no gradient from the predictor at all.
                predictability_loss_a: torch.Tensor = criterion_prediction(pred_repr_a.detach(), real_repr_a, target_ones)
                predictability_loss_b: torch.Tensor = criterion_prediction(pred_repr_b.detach(), real_repr_b, target_ones)
                predictability_loss = predictability_loss_a + predictability_loss_b

                # 4b. KL-style loss to prevent representation collapse
                all_real_reprs = torch.cat([real_repr_a, real_repr_b], dim=0)  # [2*batch, response_repr_dim]
                repr_kl_loss: torch.Tensor = self._compute_repr_kl_loss(all_real_reprs)

                # 5. Mix representations based on current ratio
                use_real = torch.rand(batch_size, device=self.device, generator=generator) < current_ratio  # [batch]
                repr_a = torch.where(use_real.unsqueeze(-1), real_repr_a, pred_repr_a)  # [batch, response_repr_dim]
                repr_b = torch.where(use_real.unsqueeze(-1), real_repr_b, pred_repr_b)  # [batch, response_repr_dim]

                # 6. Score (mixed) representations
                score_a = self.network.forward_score(
                    batch_prompt_emb_a,
                    batch_prompt_feat_a,
                    repr_a,
                )  # [batch]
                score_b = self.network.forward_score(
                    batch_prompt_emb_b,
                    batch_prompt_feat_b,
                    repr_b,
                )  # [batch]
                scoring_loss: torch.Tensor = criterion_ranking(score_a, score_b, batch_labels)

                # 7. Total loss
                total_batch_loss = (
                    scoring_loss
                    + self.prediction_loss_weight * pred_loss
                    + self.predictability_loss_weight * predictability_loss
                    + self.repr_kl_loss_weight * repr_kl_loss
                )
                total_batch_loss.backward()
                optimizer.step()

                # Compute metrics
                total_loss += total_batch_loss.item()
                total_scoring_loss += scoring_loss.item()
                total_prediction_loss += pred_loss.item()
                total_predictability_loss += predictability_loss.item()
                total_repr_kl_loss += repr_kl_loss.item()

                with torch.no_grad():
                    metrics = self._compute_batch_metrics(
                        pred_repr_a, pred_repr_b,
                        real_repr_a, real_repr_b,
                        score_a, score_b,
                        batch_prompt_emb_a, batch_prompt_feat_a,
                        batch_prompt_emb_b, batch_prompt_feat_b,
                        batch_labels,
                    )
                    total_train_accuracy += metrics["accuracy"]
                    total_prediction_quality += metrics["prediction_quality"]
                    total_scorer_real_repr_accuracy += metrics["scorer_real_repr_accuracy"]
                    total_repr_variance += metrics["repr_variance"]

                n_batches += 1

            # Average metrics
            avg_loss = total_loss / n_batches
            avg_scoring_loss = total_scoring_loss / n_batches
            avg_prediction_loss = total_prediction_loss / n_batches
            avg_predictability_loss = total_predictability_loss / n_batches
            avg_repr_kl_loss = total_repr_kl_loss / n_batches
            avg_train_accuracy = total_train_accuracy / n_batches
            avg_prediction_quality = total_prediction_quality / n_batches
            avg_scorer_real_repr_accuracy = total_scorer_real_repr_accuracy / n_batches
            avg_repr_variance = total_repr_variance / n_batches

            # Validation
            with Timer("perform_validation", verbosity="start+end", parent=timer):
                val_metrics = self._perform_validation(
                    val_dataloader,
                    criterion_ranking,
                    criterion_prediction,
                    current_ratio,
                    epoch,
                    timer,
                ) if val_dataloader is not None else None

            # Create training history entry
            additional_metrics = {
                "scoring_loss": avg_scoring_loss,
                "prediction_loss": avg_prediction_loss,
                "predictability_loss": avg_predictability_loss,
                "repr_kl_loss": avg_repr_kl_loss,
                "prediction_quality": avg_prediction_quality,
                "scorer_real_repr_accuracy": avg_scorer_real_repr_accuracy,
                "repr_mean_variance": avg_repr_variance,
                "current_real_repr_ratio": current_ratio,
            }

            if val_metrics is not None:
                additional_metrics.update({
                    "val_prediction_quality": val_metrics["prediction_quality"],
                    "val_scorer_real_repr_accuracy": val_metrics["scorer_real_repr_accuracy"],
                    "val_repr_mean_variance": val_metrics["repr_mean_variance"],
                    "val_predictability_loss": val_metrics["predictability_loss"],
                    "val_repr_kl_loss": val_metrics["repr_kl_loss"],
                })

            entry = TrainingHistoryEntry(
                epoch=epoch,
                total_loss=avg_loss,
                val_loss=val_metrics["loss"] if val_metrics is not None else None,
                train_accuracy=avg_train_accuracy,
                val_accuracy=val_metrics["accuracy"] if val_metrics is not None else None,
                additional_metrics=additional_metrics,
            )
            self._history_entries.append(entry)

            self.append_entry_to_log(entry, log_timings_from=self.last_timer)

        return self.EpochResult(
            epoch=epoch,
            total_loss=avg_loss,
            train_accuracy=avg_train_accuracy,
            val_loss=val_metrics["loss"] if val_metrics is not None else None,
            val_accuracy=val_metrics["accuracy"] if val_metrics is not None else None,
            duration=timer.elapsed_time,
            additional_metrics=additional_metrics,
        )

    def _perform_validation(
        self,
        val_dataloader: _DataLoaderType,
        criterion_ranking: nn.Module,
        criterion_prediction: nn.Module,
        current_ratio: float,
        epoch: int,
        timer: Timer,
    ) -> dict[str, float]:
        """Perform validation."""
        self.network.eval()

        # Create seeded generator for reproducible mixing (use different seed for validation)
        generator = torch.Generator(device=self.device).manual_seed(self.seed + epoch + 10000)

        total_loss = 0.0
        total_accuracy = 0.0
        total_prediction_quality = 0.0
        total_scorer_real_repr_accuracy = 0.0
        total_repr_variance = 0.0
        total_predictability_loss = 0.0
        total_repr_kl_loss = 0.0
        n_batches = 0

        for (
            batch_prompt_emb_a,
            batch_prompt_feat_a,
            batch_model_emb_a,
            batch_response_emb_a,
            batch_response_feat_a,
            batch_prompt_emb_b,
            batch_prompt_feat_b,
            batch_model_emb_b,
            batch_response_emb_b,
            batch_response_feat_b,
            batch_labels,
        ) in val_dataloader:
            # Move to device
            batch_prompt_emb_a = batch_prompt_emb_a.to(self.device)  # [batch, prompt_embedding_dim]
            batch_prompt_feat_a = batch_prompt_feat_a.to(self.device)  # [batch, prompt_features_dim]
            batch_model_emb_a = batch_model_emb_a.to(self.device)  # [batch, model_embedding_dim]
            batch_response_emb_a = batch_response_emb_a.to(self.device)  # [batch, response_embedding_dim]
            batch_response_feat_a = batch_response_feat_a.to(self.device)  # [batch, response_features_dim]

            batch_prompt_emb_b = batch_prompt_emb_b.to(self.device)  # [batch, prompt_embedding_dim]
            batch_prompt_feat_b = batch_prompt_feat_b.to(self.device)  # [batch, prompt_features_dim]
            batch_model_emb_b = batch_model_emb_b.to(self.device)  # [batch, model_embedding_dim]
            batch_response_emb_b = batch_response_emb_b.to(self.device)  # [batch, response_embedding_dim]
            batch_response_feat_b = batch_response_feat_b.to(self.device)  # [batch, response_features_dim]

            batch_labels = batch_labels.to(self.device)  # [batch]
            batch_size = len(batch_labels)

            with torch.no_grad():
                # Predict response representations
                pred_repr_a = self.network.forward_predict(
                    batch_prompt_emb_a,
                    batch_prompt_feat_a,
                    batch_model_emb_a,
                )  # [batch, response_repr_dim]
                pred_repr_b = self.network.forward_predict(
                    batch_prompt_emb_b,
                    batch_prompt_feat_b,
                    batch_model_emb_b,
                )  # [batch, response_repr_dim]

                # Encode real responses
                real_repr_a = self.network.forward_encode_response(
                    batch_response_emb_a,
                    batch_response_feat_a,
                )  # [batch, response_repr_dim]
                real_repr_b = self.network.forward_encode_response(
                    batch_response_emb_b,
                    batch_response_feat_b,
                )  # [batch, response_repr_dim]

                # Prediction loss (predictor toward encoder)
                target_ones = torch.ones(batch_size, device=self.device)  # [batch]
                pred_loss_a = criterion_prediction(pred_repr_a, real_repr_a, target_ones)
                pred_loss_b = criterion_prediction(pred_repr_b, real_repr_b, target_ones)
                pred_loss = pred_loss_a + pred_loss_b

                # Predictability loss (encoder toward predictor)
                predictability_loss_a = criterion_prediction(pred_repr_a, real_repr_a, target_ones)
                predictability_loss_b = criterion_prediction(pred_repr_b, real_repr_b, target_ones)
                predictability_loss = predictability_loss_a + predictability_loss_b

                # KL-style loss
                all_real_reprs = torch.cat([real_repr_a, real_repr_b], dim=0)  # [2*batch, response_repr_dim]
                repr_kl_loss = self._compute_repr_kl_loss(all_real_reprs)

                # Mix representations (same as training)
                use_real = torch.rand(batch_size, device=self.device, generator=generator) < current_ratio  # [batch]
                repr_a = torch.where(use_real.unsqueeze(-1), real_repr_a, pred_repr_a)  # [batch, response_repr_dim]
                repr_b = torch.where(use_real.unsqueeze(-1), real_repr_b, pred_repr_b)  # [batch, response_repr_dim]

                # Score
                score_a = self.network.forward_score(
                    batch_prompt_emb_a,
                    batch_prompt_feat_a,
                    repr_a,
                )  # [batch]
                score_b = self.network.forward_score(
                    batch_prompt_emb_b,
                    batch_prompt_feat_b,
                    repr_b,
                )  # [batch]
                scoring_loss = criterion_ranking(score_a, score_b, batch_labels)

                # Total loss
                batch_loss = (
                    scoring_loss
                    + self.prediction_loss_weight * pred_loss
                    + self.predictability_loss_weight * predictability_loss
                    + self.repr_kl_loss_weight * repr_kl_loss
                )
                total_loss += batch_loss.item()
                total_predictability_loss += predictability_loss.item()
                total_repr_kl_loss += repr_kl_loss.item()

                # Compute metrics using shared function
                metrics = self._compute_batch_metrics(
                    pred_repr_a, pred_repr_b,
                    real_repr_a, real_repr_b,
                    score_a, score_b,
                    batch_prompt_emb_a, batch_prompt_feat_a,
                    batch_prompt_emb_b, batch_prompt_feat_b,
                    batch_labels,
                )
                total_accuracy += metrics["accuracy"]
                total_prediction_quality += metrics["prediction_quality"]
                total_scorer_real_repr_accuracy += metrics["scorer_real_repr_accuracy"]
                total_repr_variance += metrics["repr_variance"]

            n_batches += 1

        return {
            "loss": total_loss / n_batches,
            "accuracy": total_accuracy / n_batches,
            "prediction_quality": total_prediction_quality / n_batches,
            "scorer_real_repr_accuracy": total_scorer_real_repr_accuracy / n_batches,
            "repr_mean_variance": total_repr_variance / n_batches,
            "predictability_loss": total_predictability_loss / n_batches,
            "repr_kl_loss": total_repr_kl_loss / n_batches,
        }

    @staticmethod
    def _compute_repr_kl_loss(reprs: torch.Tensor) -> torch.Tensor:
        """
        Compute a KL-style loss to prevent representation collapse.

        Treats the batch of representations as samples from a per-dimension Gaussian
        and computes KL(N(μ_batch, σ²_batch) || N(0, 1)):
            KL = 0.5 * mean(σ² + μ² - 1 - log(σ²))

        This penalises both mean shift and variance collapse/explosion, with the
        log term providing a strong gradient when variance approaches zero.
        """
        mu = reprs.mean(dim=0)  # [response_repr_dim]
        var = reprs.var(dim=0).clamp(min=1e-8)  # [response_repr_dim]
        kl = 0.5 * (var + mu.pow(2) - 1.0 - var.log())  # [response_repr_dim]
        return kl.mean()

    def _compute_batch_metrics(
        self,
        pred_repr_a: torch.Tensor,  # [batch, response_repr_dim]
        pred_repr_b: torch.Tensor,  # [batch, response_repr_dim]
        real_repr_a: torch.Tensor,  # [batch, response_repr_dim]
        real_repr_b: torch.Tensor,  # [batch, response_repr_dim]
        score_a: torch.Tensor,  # [batch]
        score_b: torch.Tensor,  # [batch]
        prompt_emb_a: torch.Tensor,  # [batch, prompt_embedding_dim]
        prompt_feat_a: torch.Tensor,  # [batch, prompt_features_dim]
        prompt_emb_b: torch.Tensor,  # [batch, prompt_embedding_dim]
        prompt_feat_b: torch.Tensor,  # [batch, prompt_features_dim]
        labels: torch.Tensor,  # [batch]
    ) -> dict[str, float]:
        """
        Compute batch metrics for training/validation.
        
        Returns:
            Dictionary with accuracy, prediction_quality, scorer_real_repr_accuracy, repr_variance
        """
        # Accuracy (using mixed/predicted representations)
        accuracy = compute_pairwise_accuracy(score_a, score_b, labels)

        # Prediction quality (cosine similarity between predicted and real)
        cos_sim_a = F.cosine_similarity(pred_repr_a, real_repr_a, dim=1)  # [batch]
        cos_sim_b = F.cosine_similarity(pred_repr_b, real_repr_b, dim=1)  # [batch]
        prediction_quality = ((cos_sim_a.mean() + cos_sim_b.mean()) / 2 + 1) / 2  # rescale to [0, 1]

        # Scorer real representation accuracy
        real_score_a = self.network.forward_score(prompt_emb_a, prompt_feat_a, real_repr_a)  # [batch]
        real_score_b = self.network.forward_score(prompt_emb_b, prompt_feat_b, real_repr_b)  # [batch]
        scorer_real_repr_accuracy = compute_pairwise_accuracy(real_score_a, real_score_b, labels)

        # Representation variance
        all_reprs = torch.cat([real_repr_a, real_repr_b], dim=0)  # [2*batch, response_repr_dim]
        repr_variance = all_reprs.var(dim=0).mean().item()

        return {
            "accuracy": accuracy,
            "prediction_quality": prediction_quality.item(),
            "scorer_real_repr_accuracy": scorer_real_repr_accuracy,
            "repr_variance": repr_variance,
        }

    def _log_epoch_result(self, result: "ResponsePredictiveModel.EpochResult") -> None:
        """Log epoch results."""
        if self.print_every is None:
            return

        if not result.epoch % self.print_every == 0:
            return

        # Extract additional metrics
        metrics = result.additional_metrics
        pred_quality = metrics.get("prediction_quality", 0.0)
        scorer_real_acc = metrics.get("scorer_real_repr_accuracy", 0.0)
        repr_var = metrics.get("repr_mean_variance", 0.0)
        current_ratio = metrics.get("current_real_repr_ratio", 0.0)

        if result.val_loss is None or result.val_accuracy is None:
            print(
                f"Epoch {result.epoch:>4}: loss = {result.total_loss:.4f}, "
                f"acc = {(result.train_accuracy*100):.2f}%, "
                f"pred_q = {pred_quality:.3f}, scorer_real_acc = {(scorer_real_acc*100):.2f}%, "
                f"repr_var = {repr_var:.4f}, ratio = {current_ratio:.2f} - {result.duration:.2f}s"
            )
        else:
            val_pred_quality = metrics.get("val_prediction_quality", 0.0)
            val_scorer_real_acc = metrics.get("val_scorer_real_repr_accuracy", 0.0)
            print(
                f"Epoch {result.epoch:>4}: loss = {result.total_loss:.4f}/{result.val_loss:.4f}, "
                f"acc = {(result.train_accuracy*100):.2f}%/{(result.val_accuracy*100):.2f}%, "
                f"pred_q = {pred_quality:.3f}/{val_pred_quality:.3f}, "
                f"scorer_real_acc = {(scorer_real_acc*100):.2f}%/{(val_scorer_real_acc*100):.2f}%, "
                f"repr_var = {repr_var:.4f}, ratio = {current_ratio:.2f} - {result.duration:.2f}s"
            )

    @dataclass
    class EpochResult:
        """Results from one epoch of training."""
        epoch: int
        total_loss: float
        train_accuracy: float
        val_loss: float | None
        val_accuracy: float | None
        duration: float
        additional_metrics: dict[str, float]

    class _ResponsePredictiveNetwork(nn.Module):
        """Neural network with three components: ResponseEncoder, ResponsePredictor, ResponseScorer."""

        def __init__(
            self,
            prompt_embedding_dim: int,
            prompt_features_dim: int,
            model_embedding_dim: int,
            response_embedding_dim: int,
            response_features_dim: int,
            response_repr_dim: int,
            encoder_hidden_dims: list[int],
            predictor_hidden_dims: list[int],
            scorer_hidden_dims: list[int],
            dropout: float,
        ) -> None:
            super().__init__()

            self.response_encoder = self._create_response_encoder(
                response_embedding_dim,
                response_features_dim,
                response_repr_dim,
                encoder_hidden_dims,
                dropout,
            )

            self.response_predictor = self._create_response_predictor(
                prompt_embedding_dim,
                prompt_features_dim,
                model_embedding_dim,
                response_repr_dim,
                predictor_hidden_dims,
                dropout,
            )

            self.response_scorer = self._create_response_scorer(
                prompt_embedding_dim,
                prompt_features_dim,
                response_repr_dim,
                scorer_hidden_dims,
                dropout,
            )

        @staticmethod
        def _create_response_encoder(
            response_embedding_dim: int,
            response_features_dim: int,
            response_repr_dim: int,
            hidden_dims: list[int],
            dropout: float,
        ) -> nn.Sequential:
            """Create ResponseEncoder: (response_embedding + response_features) -> response_repr_dim."""
            layers = []
            input_dim = response_embedding_dim + response_features_dim
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.1))
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, response_repr_dim))
            return nn.Sequential(*layers)

        @staticmethod
        def _create_response_predictor(
            prompt_embedding_dim: int,
            prompt_features_dim: int,
            model_embedding_dim: int,
            response_repr_dim: int,
            hidden_dims: list[int],
            dropout: float,
        ) -> nn.Sequential:
            """Create ResponsePredictor: (prompt_embedding + prompt_features + model_embedding) -> response_repr_dim."""
            layers = []
            input_dim = prompt_embedding_dim + prompt_features_dim + model_embedding_dim
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.1))
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, response_repr_dim))
            return nn.Sequential(*layers)

        @staticmethod
        def _create_response_scorer(
            prompt_embedding_dim: int,
            prompt_features_dim: int,
            response_repr_dim: int,
            hidden_dims: list[int],
            dropout: float,
        ) -> nn.Sequential:
            """Create ResponseScorer: (prompt_embedding + prompt_features + response_repr) -> 1."""
            layers = []
            input_dim = prompt_embedding_dim + prompt_features_dim + response_repr_dim
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.1))
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, 1))
            return nn.Sequential(*layers)

        def forward_predict(
            self,
            prompt_embedding: torch.Tensor,  # [batch, d_prompt_emb]
            prompt_features: torch.Tensor,  # [batch, d_prompt_features]
            model_embedding: torch.Tensor,  # [batch, d_model_emb]
        ) -> torch.Tensor:
            """Predict response representation."""
            combined = torch.cat([prompt_embedding, prompt_features, model_embedding], dim=1)  # [batch, d_prompt_emb + d_prompt_features + d_model_emb]
            return self.response_predictor(combined)  # [batch, response_repr_dim]

        def forward_encode_response(
            self,
            response_embedding: torch.Tensor,  # [batch, d_response_emb]
            response_features: torch.Tensor,  # [batch, d_response_features]
        ) -> torch.Tensor:
            """Encode real response to response representation."""
            combined = torch.cat([response_embedding, response_features], dim=1)  # [batch, d_response_emb + d_response_features]
            return self.response_encoder(combined)  # [batch, response_repr_dim]

        def forward_score(
            self,
            prompt_embedding: torch.Tensor,  # [batch, d_prompt_emb]
            prompt_features: torch.Tensor,  # [batch, d_prompt_features]
            response_repr: torch.Tensor,  # [batch, response_repr_dim]
        ) -> torch.Tensor:
            """Score a response representation in context of a prompt."""
            combined = torch.cat([prompt_embedding, prompt_features, response_repr], dim=1)  # [batch, d_prompt_emb + d_prompt_features + response_repr_dim]
            output: torch.Tensor = self.response_scorer(combined)  # [batch, 1]
            score = output.squeeze(-1)  # [batch]
            return torch.tanh(score)  # [batch] - constrained to [-1, 1]

        def forward(
            self,
            prompt_embedding: torch.Tensor,  # [batch, d_prompt_emb]
            prompt_features: torch.Tensor,  # [batch, d_prompt_features]
            model_embedding: torch.Tensor,  # [batch, d_model_emb]
        ) -> torch.Tensor:
            """Full forward pass for inference: predict then score."""
            predicted_repr = self.forward_predict(prompt_embedding, prompt_features, model_embedding)  # [batch, response_repr_dim]
            return self.forward_score(prompt_embedding, prompt_features, predicted_repr)  # [batch]
