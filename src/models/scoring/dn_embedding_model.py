"""Dense network model for prompt routing."""

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter
from pydantic import TypeAdapter

from src.models.scoring.scoring_model_base import ScoringModelBase
from src.data_models.data_models import TrainingData, InputData, OutputData
from src.data_models.dn_embedding_network_types import PreprocessedPromptPair, PreprocessedTrainingData, PromptRoutingOutput
from src.models.embedding_specs.embedding_spec_union import EmbeddingSpec
from src.models.embedding_models.embedding_model_base import EmbeddingModelBase
from src.models.optimizers.adamw_spec import AdamWSpec
from src.preprocessing.prompt_embedding_preprocessor import PromptEmbeddingPreprocessor
from src.preprocessing.simple_scaler import SimpleScaler
from src.utils.string_encoder import StringEncoder
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.timer import Timer
from src.utils.torch_utils import state_dict_to_cpu
from src.utils.accuracy import compute_pairwise_accuracy
from src.utils.data_split import ValidationSplit, split_dn_embedding_preprocessed_data
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models import model_loading
from src.models.model_outputs_cache import ModelOutputsCache
from src.utils.best_model_tracker import BestModelTracker
from src.utils.ranking_loss import PairwiseRankingLossType, compute_pairwise_ranking_loss
from src.analysis.training_diagnostics import EpochDiagnosticsAccumulator, split_tensor_with_grad
from src.preprocessing.scoring_feature_extraction import get_feature_descriptions


_DataLoaderType = DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


class DnEmbeddingModel(ScoringModelBase):
    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        use_skip_connections: bool = False,
        input_proj_dim: int = 64,
        optimizer_spec: OptimizerSpecification | None = None,
        balance_model_samples: bool = True,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_spec: EmbeddingSpec | None = None,
        load_embedding_model_from: str | None = None,
        min_model_comparisons: int = 20,
        embedding_model_epochs: int = 10,
        base_model_name: str | None = None,
        run_name: str | None = None,
        print_every: int | None = None,
        seed: int = 42,
        ranking_loss_type: PairwiseRankingLossType = "margin_ranking",
    ) -> None:
        super().__init__(run_name)

        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 128, 64]
        self.ranking_loss_type = ranking_loss_type
        self.dropout = dropout
        self.use_skip_connections = use_skip_connections
        self.input_proj_dim = input_proj_dim
        self.optimizer_spec = optimizer_spec if optimizer_spec is not None else AdamWSpec(learning_rate=0.001)
        self.balance_model_samples = balance_model_samples
        self.embedding_model_name = embedding_model_name
        self.print_every = print_every
        self.min_model_comparisons = min_model_comparisons
        self.embedding_model_epochs = embedding_model_epochs
        self.seed = seed
        
        self.embedding_spec = embedding_spec
        self.embedding_model: EmbeddingModelBase | None = None
        
        self.preprocessor = PromptEmbeddingPreprocessor(
            embedding_model_name=embedding_model_name,
            min_model_comparisons=min_model_comparisons,
        )
        
        self._base_model_name: str | None = base_model_name
        self._model_outputs_cache: ModelOutputsCache | None = None
        
        self._embedding_model_source: str | None = load_embedding_model_from
        self._prompt_embedding_dim: int | None = None
        self._prompt_features_dim: int | None = None
        self._network: DnEmbeddingModel._DenseNetwork | None = None
        self._history_entries: list[TrainingHistoryEntry] = []
        self._optimizer_state: dict[str, Any] | None = None
        self._scheduler_state: dict[str, Any] | None = None
        self._epochs_completed: int = 0
        self._prompt_features_scaler: SimpleScaler | None = None
        self._best_model_tracker = BestModelTracker()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.last_timer: Timer | None = None

    @property
    def network(self) -> "_DenseNetwork":
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
    ) -> None:
        self._prompt_embedding_dim = prompt_embedding_dim
        self._prompt_features_dim = prompt_features_dim
        self._network = self._DenseNetwork(
            prompt_embedding_dim=prompt_embedding_dim,
            prompt_features_dim=prompt_features_dim,
            model_embedding_dim=self.embedding_model.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            use_skip_connections=self.use_skip_connections,
            input_proj_dim=self.input_proj_dim,
        ).to(self.device)

    def get_config_for_logging(self) -> dict[str, Any]:
        """Get configuration dictionary for Weights & Biases logging."""
        return {
            "model_type": "dn_embedding",
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "use_skip_connections": self.use_skip_connections,
            "input_proj_dim": self.input_proj_dim,
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "balance_model_samples": self.balance_model_samples,
            "embedding_model_name": self.embedding_model_name,
            "preprocessor_version": self.preprocessor.version,
            "embedding_type": self.embedding_model.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump() if self.embedding_spec is not None else None,
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "ranking_loss_type": self.ranking_loss_type,
            "base_model": self._base_model_name,
        }

    def train(
        self,
        data: TrainingData,
        validation_split: ValidationSplit | None = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        with Timer("train", verbosity="start+end") as train_timer:
            self.last_timer = train_timer
            
            with Timer("load_base_model", verbosity="start+end", parent=train_timer):
                self._load_base_model()
            
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
            
            self.init_logger_if_needed()
            self.embedding_model.set_training_logger(self._logger)

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
            
            with Timer("encode_prompts", verbosity="start+end", parent=train_timer):
                encoded_prompts = self.preprocessor.preprocess(data)

            self._prompt_features_scaler = SimpleScaler.from_state_dict(encoded_prompts.scaler_state)
            if self._network is None:
                self._initialize_network(
                    prompt_embedding_dim=encoded_prompts.embedding_dim,
                    prompt_features_dim=encoded_prompts.prompt_features_dim,
                )
            
            with Timer("cache_base_model_predictions", verbosity="start+end", parent=train_timer):
                if self._model_outputs_cache is not None:
                    self._model_outputs_cache.compute_and_cache(
                        entries=[data.entries[i] for i in encoded_prompts.filtered_indexes],
                        indexes=encoded_prompts.filtered_indexes,
                        timer=train_timer,
                    )
            
            with Timer("prepare_preprocessed_data", verbosity="start+end", parent=train_timer):
                if self._model_outputs_cache is not None:
                    base_scores_a, base_scores_b = self._model_outputs_cache.get_base_scores(encoded_prompts.filtered_indexes)
                else:
                    base_scores_a = [0.0] * len(encoded_prompts.pairs)
                    base_scores_b = [0.0] * len(encoded_prompts.pairs)
                
                preprocessed_pairs = [
                    PreprocessedPromptPair(
                        prompt_embedding=pair.prompt_embedding,
                        prompt_features=pair.prompt_features,
                        model_embedding_a=self.model_embeddings[encoded_prompts.model_encoder.decode(pair.model_id_a)],
                        model_embedding_b=self.model_embeddings[encoded_prompts.model_encoder.decode(pair.model_id_b)],
                        model_id_a=pair.model_id_a,
                        model_id_b=pair.model_id_b,
                        winner_label=pair.winner_label,
                        base_score_a=base_a,
                        base_score_b=base_b,
                    )
                    for pair, base_a, base_b in zip(encoded_prompts.pairs, base_scores_a, base_scores_b)
                ]
                preprocessed_data = PreprocessedTrainingData(
                    pairs=preprocessed_pairs,
                    prompt_features_dim=encoded_prompts.prompt_features_dim,
                )
            
            with Timer("split_preprocessed_data", verbosity="start+end", parent=train_timer):
                preprocessed_train, preprocessed_val = split_dn_embedding_preprocessed_data(
                    preprocessed_data,
                    val_fraction=validation_split.val_fraction if validation_split is not None else 0,
                    seed=validation_split.seed if validation_split is not None else 42,
                )
            
            with Timer("prepare_dataloaders", verbosity="start+end", parent=train_timer):
                dataloader = self._prepare_dataloader(preprocessed_train, batch_size, use_balancing=True)
                val_dataloader = self._prepare_dataloader(preprocessed_val, batch_size, use_balancing=False) if preprocessed_val is not None else None
            
            optimizer, scheduler = self._create_optimizer_and_scheduler()

            with Timer("epochs", verbosity="start+end", parent=train_timer) as epochs_timer:
                for epoch in range(self._epochs_completed + 1, self._epochs_completed + epochs + 1):
                    result = self._train_epoch(epoch, dataloader, val_dataloader, optimizer, epochs_timer)
                    
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

            with Timer("sensitivity_analysis", verbosity="start+end", parent=train_timer):
                sensitivity_metrics = DnEmbeddingModel._SensitivityAnalysis.compute(
                    self.network,
                    train_dataloader=dataloader,
                    val_dataloader=val_dataloader,
                    seed=self.seed,
                )

            final_metrics = {
                "best_epoch": self._best_model_tracker.best_epoch,
                "best_accuracy": self._best_model_tracker.best_accuracy,
                "total_epochs": self._epochs_completed,
                **sensitivity_metrics,
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
            
            with Timer("predict_base_model", verbosity="start+end", parent=predict_timer):
                base_result = self._model_outputs_cache.predict(X, batch_size=batch_size) \
                    if self._model_outputs_cache is not None else None
            
            with Timer("preprocess_input", verbosity="start+end", parent=predict_timer):
                encoded_prompts = self.preprocessor.preprocess_for_inference(
                    prompts=X.prompts,
                    model_names=X.model_names,
                    model_encoder=StringEncoder(), # We don't need model IDs here
                    scaler=self._prompt_features_scaler,
                )
            
            prompt_embeddings = torch.from_numpy(encoded_prompts.prompt_embeddings).to(self.device)  # [n_prompts, embedding_dim]
            prompt_features = torch.from_numpy(encoded_prompts.prompt_features).to(self.device)  # [n_prompts, prompt_features_dim]
            mean_model_embedding = np.mean(list(self.model_embeddings.values()), axis=0)  # [model_embedding_dim]
            model_embeddings = torch.from_numpy(np.array([
                self.model_embeddings.get(model_name, mean_model_embedding)
                for model_name in X.model_names
            ])).to(self.device) # [n_models, model_embedding_dim]
            
            self.network.eval()
            scores_dict: dict[str, np.ndarray] = {}
            
            with torch.no_grad():
                for model_idx, model_name in enumerate(X.model_names):
                    model_emb = model_embeddings[model_idx]  # [model_embedding_dim]
                    model_scores = []
                    
                    for i in range(0, len(prompt_embeddings), batch_size):
                        batch_embeddings = prompt_embeddings[i:i + batch_size]  # [batch_size, embedding_dim]
                        batch_features = prompt_features[i:i + batch_size]  # [batch_size, prompt_features_dim]
                        batch_size_actual = len(batch_embeddings)

                        batch_model_embs = model_emb.unsqueeze(0).expand(batch_size_actual, -1)  # [batch_size, model_embedding_dim]

                        batch_scores = self.network(
                            batch_embeddings,
                            batch_features,
                            batch_model_embs,
                        )  # [batch_size]
                        
                        # Apply tanh to constrain to [-1, 1]
                        batch_scores = torch.tanh(batch_scores)  # [batch_size]
                        model_scores.append(batch_scores)
                    
                    all_scores = torch.cat(model_scores)  # [n_prompts]
                    scores_dict[model_name] = all_scores.cpu().numpy()
            
            if base_result is not None:
                scores_dict = {
                    model_name: scores + base_result.scores.get(model_name, np.zeros_like(scores))
                    for model_name, scores in scores_dict.items()
                }
            
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
        if self._network is None or self._prompt_features_scaler is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        if not self.embedding_model.is_initialized:
            raise RuntimeError("Embedding model not initialized")
        
        return {
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "optimizer_state": state_dict_to_cpu(self._optimizer_state) if self._optimizer_state is not None else None,
            "scheduler_state": state_dict_to_cpu(self._scheduler_state) if self._scheduler_state is not None else None,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "use_skip_connections": self.use_skip_connections,
            "input_proj_dim": self.input_proj_dim,
            "balance_model_samples": self.balance_model_samples,
            "embedding_model_name": self.embedding_model_name,
            "print_every": self.print_every,
            "preprocessor_version": self.preprocessor.version,
            "prompt_embedding_dim": self._prompt_embedding_dim,
            "prompt_features_dim": self._prompt_features_dim,
            "network_state_dict": state_dict_to_cpu(self.network.state_dict()),
            "epochs_completed": self._epochs_completed,
            "prompt_features_scaler_state": self._prompt_features_scaler.get_state_dict(),

            "embedding_type": self.embedding_model.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump() if self.embedding_spec is not None else None,
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "embedding_model_state_dict": self.embedding_model.get_state_dict(),
            "seed": self.seed,
            "ranking_loss_type": self.ranking_loss_type,
            "base_model_name": self._base_model_name,
            "base_model_state_dict": self._model_outputs_cache.model.get_state_dict() if self._model_outputs_cache is not None else None,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any], instance: "DnEmbeddingModel | None" = None) -> "DnEmbeddingModel":
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
                hidden_dims=state_dict["hidden_dims"],
                dropout=state_dict.get("dropout", 0.2),
                use_skip_connections=state_dict.get("use_skip_connections", False),
                input_proj_dim=state_dict.get("input_proj_dim", 64),
                optimizer_spec=optimizer_spec,
                balance_model_samples=state_dict["balance_model_samples"],
                embedding_model_name=state_dict["embedding_model_name"],
                embedding_spec=embedding_spec,
                min_model_comparisons=state_dict["min_model_comparisons"],
                embedding_model_epochs=state_dict["embedding_model_epochs"],
                base_model_name=state_dict.get("base_model_name", None),
                print_every=state_dict["print_every"],
                seed=state_dict["seed"],
                ranking_loss_type=state_dict.get("ranking_loss_type", "margin_ranking"),
            )
        
        model.embedding_model = EmbeddingModelBase.load_from_state_dict(state_dict["embedding_model_state_dict"])
        model._prompt_features_scaler = SimpleScaler.from_state_dict(state_dict["prompt_features_scaler_state"])

        if model._network is None:
            model._initialize_network(
                prompt_embedding_dim=state_dict["prompt_embedding_dim"],
                prompt_features_dim=state_dict["prompt_features_dim"],
            )
        model.network.load_state_dict(state_dict["network_state_dict"])
        model.network.to(model.device)
        
        model._optimizer_state = state_dict.get("optimizer_state")
        model._scheduler_state = state_dict.get("scheduler_state")
        model._epochs_completed = state_dict.get("epochs_completed", 0)
        
        if model._base_model_name is not None and state_dict.get("base_model_state_dict") is not None:
            model_type, _ = model._base_model_name.split("/", 1)
            base_model = model_loading.load_scoring_model_from_state_dict(model_type, state_dict["base_model_state_dict"])
            model._model_outputs_cache = ModelOutputsCache(base_model, quiet=model.print_every is None)
        
        return model

    def _load_base_model(self) -> None:
        """Load the base model from the specification string."""
        if self._base_model_name is None or self._model_outputs_cache is not None:
            return
        
        if "/" not in self._base_model_name:
            raise ValueError("Base model must be of the form 'model_type/model_name'")
        
        model_type, model_name = self._base_model_name.split("/", 1)
        
        if self.print_every is not None:
            print(f"Loading base model: {self._base_model_name}")
        
        loaded_model = model_loading.load_scoring_model(model_type, model_name)
        self._model_outputs_cache = ModelOutputsCache(
            model=loaded_model,
            quiet=self.print_every is None,
        )
        
        if self.print_every is not None:
            print("Base model loaded")

    def _prepare_dataloader(
        self, 
        preprocessed_data: PreprocessedTrainingData, 
        batch_size: int,
        use_balancing: bool,
    ) -> _DataLoaderType:
        """
        Prepare dataloader from preprocessed training data.
        
        Args:
            preprocessed_data: Preprocessed training data
            batch_size: Batch size
            use_balancing: Whether to apply sample balancing
            
        Returns:
            DataLoader for training/validation
        """
        # Prepare data for training
        # Each comparison pair becomes a training sample with margin ranking loss
        prompt_embeddings_a_list = []  # [n_pairs, embedding_dim]
        prompt_embeddings_b_list = []  # [n_pairs, embedding_dim]
        prompt_features_a_list = []  # [n_pairs, prompt_features_dim]
        prompt_features_b_list = []  # [n_pairs, prompt_features_dim]
        model_embeddings_a_list = []  # [n_pairs, model_embedding_dim]
        model_embeddings_b_list = []  # [n_pairs, model_embedding_dim]
        model_ids_a_list = []  # [n_pairs]
        model_ids_b_list = []  # [n_pairs]
        labels_list = []  # [n_pairs] - 1 if a wins, -1 if b wins
        
        base_scores_a_list = []  # [n_pairs]
        base_scores_b_list = []  # [n_pairs]
        
        for pair in preprocessed_data.pairs:
            prompt_embeddings_a_list.append(torch.from_numpy(pair.prompt_embedding))
            prompt_embeddings_b_list.append(torch.from_numpy(pair.prompt_embedding))
            prompt_features_a_list.append(torch.from_numpy(pair.prompt_features))
            prompt_features_b_list.append(torch.from_numpy(pair.prompt_features))
            model_embeddings_a_list.append(torch.from_numpy(pair.model_embedding_a))
            model_embeddings_b_list.append(torch.from_numpy(pair.model_embedding_b))
            model_ids_a_list.append(pair.model_id_a)
            model_ids_b_list.append(pair.model_id_b)
            base_scores_a_list.append(pair.base_score_a)
            base_scores_b_list.append(pair.base_score_b)
            
            # label: 1 if model_a should be ranked higher, -1 if model_b should be ranked higher
            labels_list.append(1.0 if pair.winner_label == 0 else -1.0)
        
        prompt_embeddings_a = torch.stack(prompt_embeddings_a_list)  # [n_pairs, embedding_dim]
        prompt_embeddings_b = torch.stack(prompt_embeddings_b_list)  # [n_pairs, embedding_dim]
        prompt_features_a = torch.stack(prompt_features_a_list)  # [n_pairs, prompt_features_dim]
        prompt_features_b = torch.stack(prompt_features_b_list)  # [n_pairs, prompt_features_dim]
        model_embeddings_a = torch.stack(model_embeddings_a_list)  # [n_pairs, model_embedding_dim]
        model_embeddings_b = torch.stack(model_embeddings_b_list)  # [n_pairs, model_embedding_dim]
        labels = torch.tensor(labels_list, dtype=torch.float32)  # [n_pairs]
        model_ids_a = torch.tensor(model_ids_a_list, dtype=torch.long)  # [n_pairs]
        model_ids_b = torch.tensor(model_ids_b_list, dtype=torch.long)  # [n_pairs]
        base_scores_a = torch.tensor(base_scores_a_list, dtype=torch.float32)  # [n_pairs]
        base_scores_b = torch.tensor(base_scores_b_list, dtype=torch.float32)  # [n_pairs]
        
        dataset = TensorDataset(
            prompt_embeddings_a,
            prompt_features_a,
            model_embeddings_a,
            prompt_embeddings_b,
            prompt_features_b,
            model_embeddings_b,
            labels,
            model_ids_a,
            model_ids_b,
            base_scores_a,
            base_scores_b,
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
        """
        Create a weighted sampler to balance model representation in training.
        
        For each pair, we consider both models involved. The weight for a pair
        is based on the rarest model in that pair.
        
        Args:
            model_ids_a: List of model IDs for position A  # [n_pairs]
            model_ids_b: List of model IDs for position B  # [n_pairs]
            
        Returns:
            WeightedRandomSampler that balances model representation
        """
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
        # This ensures rare models get more representation
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
        epochs_timer: Timer,
    ) -> "DnEmbeddingModel.EpochResult":
        with Timer(f"epoch_{epoch}", verbosity="start+end", parent=epochs_timer) as timer:
            self.network.train()
            total_loss = 0.0
            total_accuracy = 0.0
            n_batches = 0
            total_samples = 0
            diagnostic_accumulator = EpochDiagnosticsAccumulator()
            trunk_params, prompt_emb_proj_params, prompt_feat_proj_params, model_emb_proj_params = self.network.get_param_groups_for_metrics()
            numeric_descs, bool_descs = get_feature_descriptions()
            prompt_feature_names = [d.name for d in numeric_descs] + [d.name for d in bool_descs]

            for batch_emb_a, batch_features_a, batch_model_emb_a, batch_emb_b, batch_features_b, batch_model_emb_b, batch_labels, batch_model_ids_a, batch_model_ids_b, batch_base_scores_a, batch_base_scores_b in dataloader:
                batch_emb_a: torch.Tensor = batch_emb_a.to(self.device).requires_grad_(True)  # [batch_size, prompt_embedding_dim]
                batch_features_a: torch.Tensor = batch_features_a.to(self.device).requires_grad_(True)  # [batch_size, prompt_features_dim]
                batch_model_emb_a: torch.Tensor = batch_model_emb_a.to(self.device).requires_grad_(True)  # [batch_size, model_embedding_dim]
                batch_emb_b: torch.Tensor = batch_emb_b.to(self.device)  # [batch_size, prompt_embedding_dim]
                batch_features_b: torch.Tensor = batch_features_b.to(self.device)  # [batch_size, prompt_features_dim]
                batch_model_emb_b: torch.Tensor = batch_model_emb_b.to(self.device)  # [batch_size, model_embedding_dim]
                batch_labels: torch.Tensor = batch_labels.to(self.device)  # [batch_size]
                batch_model_ids_a: torch.Tensor = batch_model_ids_a.to(self.device)  # [batch_size]
                batch_model_ids_b: torch.Tensor = batch_model_ids_b.to(self.device)  # [batch_size]
                batch_base_scores_a: torch.Tensor = batch_base_scores_a.to(self.device)  # [batch_size]
                batch_base_scores_b: torch.Tensor = batch_base_scores_b.to(self.device)  # [batch_size]

                total_samples += len(batch_emb_a)

                with torch.no_grad():
                    pe = self.network.prompt_emb_proj(batch_emb_a)       # [batch_size, input_proj_dim]
                    pf = self.network.prompt_feat_proj(batch_features_a)  # [batch_size, input_proj_dim]
                    me = self.network.model_emb_proj(batch_model_emb_a)   # [batch_size, input_proj_dim]
                diagnostic_accumulator.update_representation_stats({
                    "prompt_emb_proj": pe,
                    "prompt_feat_proj": pf,
                    "model_emb_proj": me,
                })

                optimizer.zero_grad()
                scores_a = self.network(
                    batch_emb_a,
                    batch_features_a,
                    batch_model_emb_a,
                ) + batch_base_scores_a  # [batch_size]
                scores_b = self.network(
                    batch_emb_b,
                    batch_features_b,
                    batch_model_emb_b,
                ) + batch_base_scores_b  # [batch_size]

                loss: torch.Tensor = compute_pairwise_ranking_loss(
                    self.ranking_loss_type, scores_a, scores_b, batch_labels, margin=0.1
                )
                loss.backward()

                diagnostic_accumulator.update_grad_norms({
                    "trunk": trunk_params,
                    "prompt_emb_proj": prompt_emb_proj_params,
                    "prompt_feat_proj": prompt_feat_proj_params,
                    "model_emb_proj": model_emb_proj_params,
                })
                diagnostic_accumulator.update_gradient_attribution({
                    **split_tensor_with_grad(batch_features_a, prompt_feature_names),
                    "prompt_embedding": batch_emb_a,
                    "model_embedding": batch_model_emb_a,
                })
                diagnostic_accumulator.update_score_variance(
                    torch.cat([scores_a.detach(), scores_b.detach()]),  # [2 * batch_size]
                    torch.cat([batch_model_ids_a, batch_model_ids_b]),  # [2 * batch_size]
                )

                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.mean().item()
                
                with torch.no_grad():
                    batch_accuracy = compute_pairwise_accuracy(scores_a, scores_b, batch_labels)
                    total_accuracy += batch_accuracy
                
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            avg_accuracy = total_accuracy / n_batches
            additional_metrics = diagnostic_accumulator.to_dict()

            with Timer("perform_validation", verbosity="start+end", parent=timer):
                val_loss, val_accuracy = self._perform_validation(val_dataloader) if val_dataloader is not None else (None, None)
            
            entry = TrainingHistoryEntry(
                epoch=epoch,
                total_loss=avg_loss,
                val_loss=val_loss,
                train_accuracy=avg_accuracy,
                val_accuracy=val_accuracy,
                additional_metrics=additional_metrics,
            )
            self._history_entries.append(entry)
            
            self.append_entry_to_log(entry, log_timings_from=self.last_timer)
            
        return self.EpochResult(
            epoch=epoch,
            total_loss=avg_loss,
            train_accuracy=avg_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            duration=timer.elapsed_time,
            additional_metrics=additional_metrics,
        )

    def _perform_validation(
        self,
        val_dataloader: _DataLoaderType,
    ) -> tuple[float, float]:
        self.network.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0

        for batch_emb_a, batch_features_a, batch_model_emb_a, batch_emb_b, batch_features_b, batch_model_emb_b, batch_labels, _, _, batch_base_scores_a, batch_base_scores_b in val_dataloader:
            batch_emb_a: torch.Tensor = batch_emb_a.to(self.device)  # [batch_size, embedding_dim]
            batch_features_a: torch.Tensor = batch_features_a.to(self.device)  # [batch_size, prompt_features_dim]
            batch_model_emb_a: torch.Tensor = batch_model_emb_a.to(self.device)  # [batch_size, model_embedding_dim]
            batch_emb_b: torch.Tensor = batch_emb_b.to(self.device)  # [batch_size, prompt_embedding_dim]
            batch_features_b: torch.Tensor = batch_features_b.to(self.device)  # [batch_size, prompt_features_dim]
            batch_model_emb_b: torch.Tensor = batch_model_emb_b.to(self.device)  # [batch_size, model_embedding_dim]
            batch_labels: torch.Tensor = batch_labels.to(self.device)  # [batch_size]
            batch_base_scores_a: torch.Tensor = batch_base_scores_a.to(self.device)  # [batch_size]
            batch_base_scores_b: torch.Tensor = batch_base_scores_b.to(self.device)  # [batch_size]

            with torch.no_grad():
                scores_a = self.network(
                    batch_emb_a,
                    batch_features_a,
                    batch_model_emb_a,
                ) + batch_base_scores_a  # [batch_size]
                scores_b = self.network(
                    batch_emb_b,
                    batch_features_b,
                    batch_model_emb_b,
                ) + batch_base_scores_b  # [batch_size]

                loss: torch.Tensor = compute_pairwise_ranking_loss(
                    self.ranking_loss_type, scores_a, scores_b, batch_labels, margin=0.1
                )
                batch_accuracy = compute_pairwise_accuracy(scores_a, scores_b, batch_labels)

                total_loss += loss.mean().item()
                total_accuracy += batch_accuracy
                n_batches += 1

        avg_loss = total_loss / n_batches
        avg_accuracy = total_accuracy / n_batches
        return avg_loss, avg_accuracy

    def _log_epoch_result(self, result: "DnEmbeddingModel.EpochResult") -> None:
        if self.print_every is None:
            return
        
        if not result.epoch % self.print_every == 0:
            return
        
        if result.val_loss is None or result.val_accuracy is None:
            print(f"Epoch {result.epoch:>4}: loss = {result.total_loss:.4f}, accuracy = {(result.train_accuracy*100):.4f}% - {result.duration:.2f}s")
        else:
            print(f"Epoch {result.epoch:>4}: loss = {result.total_loss:.4f}/{result.val_loss:.4f}, accuracy = {(result.train_accuracy*100):.4f}%/{(result.val_accuracy*100):.4f}% - {result.duration:.2f}s")

    @dataclass
    class EpochResult:
        epoch: int
        total_loss: float
        train_accuracy: float
        val_loss: float | None
        val_accuracy: float | None
        duration: float
        additional_metrics: dict[str, float]

    class _SensitivityAnalysis:
        """Prompt sensitivity and feature importance metrics computed on dataloaders.

        Results are accuracy drops from baseline (positive = the modification hurts
        accuracy, i.e. the model relies on that component).

        A and B sides of each pair share the same prompt, so modifications are applied
        identically to both sides.
        """

        @classmethod
        def compute(
            cls,
            network: "DnEmbeddingModel._DenseNetwork",
            train_dataloader: "_DataLoaderType",
            val_dataloader: "_DataLoaderType | None",
            seed: int = 42,
        ) -> dict[str, float]:
            """Compute all sensitivity metrics on available dataloaders.

            Returns a flat dict of accuracy drops keyed by
            ``sensitivity/<metric>_train`` and ``sensitivity/<metric>_val``.
            Val entries are omitted when val_dataloader is None.
            """
            numeric_descs, boolean_descs = get_feature_descriptions()
            feature_names = [d.name for d in numeric_descs] + [d.name for d in boolean_descs]

            results: dict[str, float] = {}
            for suffix, dataloader in [("_train", train_dataloader), ("_val", val_dataloader)]:
                if dataloader is None:
                    continue

                global_mean_emb, global_mean_feat = cls._compute_global_means(network, dataloader)
                baseline = cls._compute_accuracy(network, dataloader)

                results[f"sensitivity/prompt{suffix}"] = baseline - cls._compute_accuracy(
                    network, dataloader,
                    lambda e, f, eb, fb: cls._shuffle_prompts(e, f, eb, fb, np.random.default_rng(seed)),
                )
                results[f"sensitivity/prompt_embedding{suffix}"] = baseline - cls._compute_accuracy(
                    network, dataloader,
                    lambda e, f, eb, fb: cls._set_emb_to_mean(e, f, eb, fb, global_mean_emb),
                )
                results[f"sensitivity/prompt_features{suffix}"] = baseline - cls._compute_accuracy(
                    network, dataloader,
                    lambda e, f, eb, fb: cls._set_feat_to_mean(e, f, eb, fb, global_mean_feat),
                )
                for idx, name in enumerate(feature_names):
                    results[f"sensitivity/feature/{name}{suffix}"] = baseline - cls._compute_accuracy(
                        network, dataloader,
                        lambda e, f, eb, fb: cls._permute_feature(e, f, eb, fb, idx, np.random.default_rng(seed)),
                    )

            return results

        @classmethod
        def _compute_accuracy(
            cls,
            network: "DnEmbeddingModel._DenseNetwork",
            dataloader: "_DataLoaderType",
            modify_prompt: Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            ] | None = None,
        ) -> float:
            """Compute accuracy on a dataloader with an optional per-batch prompt modification."""
            device = next(network.parameters()).device
            network.eval()
            total_correct = 0.0
            total_samples = 0

            with torch.no_grad():
                for (
                    prompt_emb_a, prompt_feat_a, model_emb_a,
                    prompt_emb_b, prompt_feat_b, model_emb_b,
                    labels, _, _, base_scores_a, base_scores_b,
                ) in dataloader:
                    prompt_emb_a = prompt_emb_a.to(device)   # [batch, prompt_emb_dim]
                    prompt_feat_a = prompt_feat_a.to(device)  # [batch, prompt_feat_dim]
                    model_emb_a = model_emb_a.to(device)      # [batch, model_emb_dim]
                    prompt_emb_b = prompt_emb_b.to(device)   # [batch, prompt_emb_dim]
                    prompt_feat_b = prompt_feat_b.to(device)  # [batch, prompt_feat_dim]
                    model_emb_b = model_emb_b.to(device)      # [batch, model_emb_dim]
                    labels = labels.to(device)                 # [batch]
                    base_scores_a = base_scores_a.to(device)  # [batch]
                    base_scores_b = base_scores_b.to(device)  # [batch]

                    if modify_prompt is not None:
                        prompt_emb_a, prompt_feat_a, prompt_emb_b, prompt_feat_b = modify_prompt(
                            prompt_emb_a, prompt_feat_a, prompt_emb_b, prompt_feat_b,
                        )

                    score_a = network(prompt_emb_a, prompt_feat_a, model_emb_a) + base_scores_a  # [batch]
                    score_b = network(prompt_emb_b, prompt_feat_b, model_emb_b) + base_scores_b  # [batch]

                    batch_size = labels.shape[0]
                    total_correct += compute_pairwise_accuracy(score_a, score_b, labels) * batch_size
                    total_samples += batch_size

            return total_correct / total_samples if total_samples > 0 else 0.0

        @classmethod
        def _compute_global_means(
            cls,
            network: "DnEmbeddingModel._DenseNetwork",
            dataloader: "_DataLoaderType",
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute the dataset-level mean prompt embedding and features.

            Returns:
                (mean_emb, mean_feat) shaped [1, prompt_emb_dim] and [1, prompt_feat_dim]
            """
            device = next(network.parameters()).device
            sum_emb: torch.Tensor | None = None
            sum_feat: torch.Tensor | None = None
            total = 0

            with torch.no_grad():
                for (prompt_emb_a, prompt_feat_a, _, _, _, _, _, _, _, _, _) in dataloader:
                    prompt_emb_a = prompt_emb_a.to(device)   # [batch, prompt_emb_dim]
                    prompt_feat_a = prompt_feat_a.to(device)  # [batch, prompt_feat_dim]
                    if sum_emb is None:
                        sum_emb = prompt_emb_a.sum(dim=0)
                        sum_feat = prompt_feat_a.sum(dim=0)
                    else:
                        sum_emb += prompt_emb_a.sum(dim=0)
                        sum_feat += prompt_feat_a.sum(dim=0)
                    total += prompt_emb_a.shape[0]

            return (sum_emb / total).unsqueeze(0), (sum_feat / total).unsqueeze(0)  # [1, d_emb], [1, d_feat]

        @staticmethod
        def _shuffle_prompts(
            emb_a: torch.Tensor,   # [batch, prompt_emb_dim]
            feat_a: torch.Tensor,  # [batch, prompt_feat_dim]
            emb_b: torch.Tensor,   # [batch, prompt_emb_dim]
            feat_b: torch.Tensor,  # [batch, prompt_feat_dim]
            rng: np.random.Generator,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # [batch, prompt_emb_dim], [batch, prompt_feat_dim], [batch, prompt_emb_dim], [batch, prompt_feat_dim]
            idx = torch.from_numpy(rng.permutation(emb_a.shape[0])).to(emb_a.device)
            return emb_a[idx], feat_a[idx], emb_b[idx], feat_b[idx]

        @staticmethod
        def _set_emb_to_mean(
            emb_a: torch.Tensor,    # [batch, prompt_emb_dim]
            feat_a: torch.Tensor,   # [batch, prompt_feat_dim]
            emb_b: torch.Tensor,    # [batch, prompt_emb_dim]
            feat_b: torch.Tensor,   # [batch, prompt_feat_dim]
            mean_emb: torch.Tensor, # [1, prompt_emb_dim]
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # [batch, prompt_emb_dim], [batch, prompt_feat_dim], [batch, prompt_emb_dim], [batch, prompt_feat_dim]
            return mean_emb.expand_as(emb_a), feat_a, mean_emb.expand_as(emb_b), feat_b

        @staticmethod
        def _set_feat_to_mean(
            emb_a: torch.Tensor,     # [batch, prompt_emb_dim]
            feat_a: torch.Tensor,    # [batch, prompt_feat_dim]
            emb_b: torch.Tensor,     # [batch, prompt_emb_dim]
            feat_b: torch.Tensor,    # [batch, prompt_feat_dim]
            mean_feat: torch.Tensor, # [1, prompt_feat_dim]
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # [batch, prompt_emb_dim], [batch, prompt_feat_dim], [batch, prompt_emb_dim], [batch, prompt_feat_dim]
            return emb_a, mean_feat.expand_as(feat_a), emb_b, mean_feat.expand_as(feat_b)

        @staticmethod
        def _permute_feature(
            emb_a: torch.Tensor,   # [batch, prompt_emb_dim]
            feat_a: torch.Tensor,  # [batch, prompt_feat_dim]
            emb_b: torch.Tensor,   # [batch, prompt_emb_dim]
            feat_b: torch.Tensor,  # [batch, prompt_feat_dim]
            feature_idx: int,
            rng: np.random.Generator,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # [batch, prompt_emb_dim], [batch, prompt_feat_dim], [batch, prompt_emb_dim], [batch, prompt_feat_dim]
            perm = torch.from_numpy(rng.permutation(feat_a.shape[0])).to(feat_a.device)
            feat_a = feat_a.clone()
            feat_b = feat_b.clone()
            feat_a[:, feature_idx] = feat_a[perm, feature_idx]
            feat_b[:, feature_idx] = feat_b[perm, feature_idx]
            return emb_a, feat_a, emb_b, feat_b

    class _DenseNetwork(nn.Module):

        class _ResidualBlock(nn.Module):
            def __init__(
                self,
                in_dim: int,
                out_dim: int,
                dropout: float,
                use_skip_connection: bool,
            ) -> None:
                super().__init__()
                self.linear = nn.Linear(in_dim, out_dim)
                self.norm = nn.LayerNorm(out_dim)
                self.activation = nn.LeakyReLU(0.1)
                self.dropout = nn.Dropout(dropout)
                if use_skip_connection:
                    self.skip: nn.Module = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
                self.use_skip_connection = use_skip_connection

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # [batch_size, in_dim] -> [batch_size, out_dim]
                h = self.dropout(self.activation(self.norm(self.linear(x))))  # [batch_size, out_dim]
                if self.use_skip_connection:
                    return h + self.skip(x)  # [batch_size, out_dim]
                return h  # [batch_size, out_dim]

        def __init__(
            self,
            prompt_embedding_dim: int,
            prompt_features_dim: int,
            model_embedding_dim: int,
            hidden_dims: list[int],
            dropout: float = 0.2,
            use_skip_connections: bool = False,
            input_proj_dim: int = 64,
        ) -> None:
            super().__init__()

            self.prompt_emb_proj = nn.Sequential(nn.Linear(prompt_embedding_dim, input_proj_dim), nn.LeakyReLU(0.1))
            self.prompt_feat_proj = nn.Sequential(nn.Linear(prompt_features_dim, input_proj_dim), nn.LeakyReLU(0.1))
            self.model_emb_proj = nn.Sequential(nn.Linear(model_embedding_dim, input_proj_dim), nn.LeakyReLU(0.1))

            blocks: list[nn.Module] = []
            prev_dim = input_proj_dim * 3
            for hidden_dim in hidden_dims:
                blocks.append(self._ResidualBlock(prev_dim, hidden_dim, dropout, use_skip_connections))
                prev_dim = hidden_dim
            blocks.append(nn.Linear(prev_dim, 1))
            self.trunk = nn.Sequential(*blocks)

            self._init_weights()

        def forward(
            self,
            prompt_embedding: torch.Tensor,  # [batch_size, prompt_embedding_dim]
            prompt_features: torch.Tensor,   # [batch_size, prompt_features_dim]
            model_embedding: torch.Tensor,   # [batch_size, model_embedding_dim]
        ) -> torch.Tensor:
            pe = self.prompt_emb_proj(prompt_embedding)  # [batch_size, input_proj_dim]
            pf = self.prompt_feat_proj(prompt_features)  # [batch_size, input_proj_dim]
            me = self.model_emb_proj(model_embedding)    # [batch_size, input_proj_dim]
            return self.trunk(torch.cat([pe, pf, me], dim=1)).squeeze(-1)  # [batch_size]

        def get_param_groups_for_metrics(
            self,
        ) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
            """Return (trunk_params, prompt_emb_proj_params, prompt_feat_proj_params, model_emb_proj_params) for gradient norm metrics."""
            trunk_params = list(self.trunk.parameters())
            prompt_emb_proj_params = list(self.prompt_emb_proj.parameters())
            prompt_feat_proj_params = list(self.prompt_feat_proj.parameters())
            model_emb_proj_params = list(self.model_emb_proj.parameters())
            return trunk_params, prompt_emb_proj_params, prompt_feat_proj_params, model_emb_proj_params

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

