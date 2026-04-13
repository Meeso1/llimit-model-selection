"""Dense network model for response length prediction."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pydantic import TypeAdapter

from src.models.length_prediction.length_prediction_model_base import LengthPredictionModelBase
from src.data_models.data_models import TrainingData, InputData
from src.data_models.length_prediction.length_prediction_data_models import (
    LengthPredictionOutputData,
    PreprocessedLengthPredictionTrainingDataWithEmbeddings,
)
from src.models.embedding_specs.embedding_spec_union import EmbeddingSpec
from src.models.embedding_models.embedding_model_base import EmbeddingModelBase
from src.models.optimizers.adamw_spec import AdamWSpec
from src.preprocessing.length_prediction_preprocessor import LengthPredictionPreprocessor
from src.preprocessing.simple_scaler import SimpleScaler
from src.utils.string_encoder import StringEncoder
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.timer import Timer
from src.utils.torch_utils import state_dict_to_cpu
from src.utils.data_split import ValidationSplit, split_length_prediction_preprocessed_data
from src.utils.length_prediction_metrics import compute_length_prediction_metrics
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models import model_loading
from src.utils.best_model_tracker import BestModelTracker
from src.analysis.training_diagnostics import EpochDiagnosticsAccumulator, split_tensor_with_grad
from src.preprocessing.scoring_feature_extraction import get_feature_descriptions


_DataLoaderType = DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


class DnEmbeddingLengthPredictionModel(LengthPredictionModelBase):
    """Dense network model for predicting response lengths."""
    
    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        input_proj_dim: int = 64,
        optimizer_spec: OptimizerSpecification | None = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_spec: EmbeddingSpec | None = None,
        load_embedding_model_from: str | None = None,
        min_model_comparisons: int = 20,
        embedding_model_epochs: int = 10,
        model_id_embedding_dim: int = 64,
        run_name: str | None = None,
        print_every: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(run_name)
        
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 128, 64]
        self.dropout = dropout
        self.input_proj_dim = input_proj_dim
        self.optimizer_spec = optimizer_spec if optimizer_spec is not None else AdamWSpec(learning_rate=0.001)
        self.embedding_model_name = embedding_model_name
        self.print_every = print_every
        self.min_model_comparisons = min_model_comparisons
        self.embedding_model_epochs = embedding_model_epochs
        self.model_id_embedding_dim = model_id_embedding_dim
        self.seed = seed
        
        self.embedding_spec = embedding_spec
        self.embedding_model: EmbeddingModelBase | None = None
        
        self.preprocessor = LengthPredictionPreprocessor(
            embedding_model_name=embedding_model_name,
            min_model_comparisons=min_model_comparisons,
        )
        
        self._embedding_model_source: str | None = load_embedding_model_from
        self._prompt_embedding_dim: int | None = None
        self._prompt_features_dim: int | None = None
        self._scaler: SimpleScaler | None = None
        self._network: DnEmbeddingLengthPredictionModel._DenseNetwork | None = None
        self._history_entries: list[TrainingHistoryEntry] = []
        self._optimizer_state: dict[str, Any] | None = None
        self._scheduler_state: dict[str, Any] | None = None
        self._epochs_completed: int = 0
        self._prompt_features_scaler: SimpleScaler | None = None
        self._best_model_tracker = BestModelTracker()
        self._model_avg_lengths: dict[str, float] = {}
        self._model_encoder: StringEncoder | None = None
        self._n_models: int = 0
        
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

    @property
    def scaler(self) -> SimpleScaler:
        if self._scaler is None:
            raise RuntimeError("Scaler not initialized. Train or load a model first.")
        return self._scaler

    def _initialize_network(
        self,
        prompt_embedding_dim: int,
        prompt_features_dim: int,
        n_models: int,
    ) -> None:
        self._prompt_embedding_dim = prompt_embedding_dim
        self._prompt_features_dim = prompt_features_dim
        self._n_models = n_models
        self._network = self._DenseNetwork(
            prompt_embedding_dim=prompt_embedding_dim,
            prompt_features_dim=prompt_features_dim,
            model_embedding_dim=self.embedding_model.embedding_dim,
            n_models=n_models,
            model_id_embedding_dim=self.model_id_embedding_dim,
            input_proj_dim=self.input_proj_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

    def get_config_for_logging(self) -> dict[str, Any]:
        """Get configuration dictionary for Weights & Biases logging."""
        return {
            "model_type": "dn_embedding_length_prediction",
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "input_proj_dim": self.input_proj_dim,
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "embedding_model_name": self.embedding_model_name,
            "preprocessor_version": self.preprocessor.version,
            "embedding_type": self.embedding_model.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump() if self.embedding_spec is not None else None,
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "model_id_embedding_dim": self.model_id_embedding_dim,
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
 
            self.init_logger_if_needed() # Must be called after embedding model is initialized

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
                preprocessed_without_embeddings = self.preprocessor.preprocess(data)
            
            with Timer("add_model_embeddings", verbosity="start+end", parent=train_timer):
                preprocessed_data = preprocessed_without_embeddings.add_model_embeddings(
                    self.model_embeddings,
                    self.embedding_model.embedding_dim,
                )
            
            self._scaler = SimpleScaler.from_state_dict(preprocessed_data.output_scaler_state)
            self._prompt_features_scaler = SimpleScaler.from_state_dict(preprocessed_data.prompt_features_scaler_state)
            self._model_encoder = preprocessed_data.model_encoder
                
            if self._network is None:
                self._initialize_network(
                    prompt_embedding_dim=preprocessed_data.embedding_dim,
                    prompt_features_dim=preprocessed_data.prompt_features_dim,
                    n_models=preprocessed_data.model_encoder.size,
                )
            
            with Timer("split_preprocessed_data", verbosity="start+end", parent=train_timer):
                preprocessed_train, preprocessed_val = split_length_prediction_preprocessed_data(
                    preprocessed_data,
                    val_fraction=validation_split.val_fraction if validation_split is not None else 0,
                    seed=validation_split.seed if validation_split is not None else 42,
                )
            
            self._model_avg_lengths = self._compute_model_avg_lengths(preprocessed_train)
            
            with Timer("prepare_dataloaders", verbosity="start+end", parent=train_timer):
                dataloader = self._prepare_dataloader(preprocessed_train, batch_size, self._model_avg_lengths)
                val_dataloader = self._prepare_dataloader(preprocessed_val, batch_size, self._model_avg_lengths) if preprocessed_val is not None else None
            
            optimizer, scheduler = self._create_optimizer_and_scheduler()
            
            # Use MSE loss for regression
            criterion = nn.MSELoss()
            
            with Timer("epochs", verbosity="start+end", parent=train_timer) as epochs_timer:
                for epoch in range(self._epochs_completed + 1, self._epochs_completed + epochs + 1):
                    result = self._train_epoch(epoch, dataloader, val_dataloader, optimizer, criterion, epochs_timer)
                    
                    self._log_epoch_result(result)
                    
                    # Use relative accuracy (1 - avg_relative_error) as the metric to track
                    train_rel_acc = 1 - result.train_metrics["avg_relative_error"]
                    val_rel_acc = 1 - result.val_metrics["avg_relative_error"] if result.val_metrics is not None else None
                    
                    self._best_model_tracker.record_state(
                        accuracy=val_rel_acc if val_rel_acc is not None else train_rel_acc,
                        state_dict=self.get_state_dict(),
                        epoch=epoch
                    )
                    
                    if scheduler is not None:
                        scheduler.step()
                    
                    self._epochs_completed = epoch
            
            # Revert to best model parameters if available
            if self._best_model_tracker.has_best_state:
                print(f"\nReverting to best model parameters from epoch {self._best_model_tracker.best_epoch} (relative error={(1 - self._best_model_tracker.best_accuracy):.4f})")
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
    ) -> LengthPredictionOutputData:
        """
        Predict response lengths for the given prompts and models.
        
        Args:
            X: Input data with prompts and model_names
            batch_size: Batch size for prediction
            
        Returns:
            LengthPredictionOutputData with predicted lengths for each model
        """
        if self._network is None or self._model_encoder is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        with Timer("predict", verbosity="start+end") as predict_timer:
            self.last_timer = predict_timer
            with Timer("preprocess_input", verbosity="start+end", parent=predict_timer):
                encoded_prompts = self.preprocessor.preprocess_for_inference(
                    prompts=X.prompts,
                    model_names=X.model_names,
                    model_encoder=self._model_encoder,
                    scaler=self._prompt_features_scaler,
                )
            
            prompt_embeddings = torch.from_numpy(encoded_prompts.prompt_embeddings).to(self.device)  # [n_prompts, embedding_dim]
            prompt_features = torch.from_numpy(encoded_prompts.prompt_features).to(self.device)  # [n_prompts, prompt_features_dim]

            # Map model names to IDs; unknown models use index n_models (dedicated unknown slot)
            model_id_map = dict(zip(X.model_names, encoded_prompts.model_ids))
            
            self.network.eval()
            predictions_dict: dict[str, np.ndarray] = {}
            
            with torch.no_grad():
                for model_name in X.model_names:
                    if model_name not in self.model_embeddings:
                        # Use default embedding if model not seen during training
                        model_embedding = self.model_embeddings.get("default", np.zeros(self.embedding_model.embedding_dim))
                    else:
                        model_embedding = self.model_embeddings[model_name]
                    
                    raw_model_id = model_id_map[model_name]
                    # Unknown models use -1; the network computes the mean of all known embeddings for them
                    model_id_idx = raw_model_id if raw_model_id is not None else -1

                    model_embedding_tensor = torch.from_numpy(model_embedding).to(self.device)
                    model_predictions = []
                    
                    for i in range(0, len(prompt_embeddings), batch_size):
                        batch_embeddings = prompt_embeddings[i:i + batch_size]  # [batch_size, embedding_dim]
                        batch_features = prompt_features[i:i + batch_size]  # [batch_size, prompt_features_dim]
                        batch_size_actual = len(batch_embeddings)
                        
                        # Repeat model embedding for batch
                        batch_model_embedding = model_embedding_tensor.unsqueeze(0).repeat(batch_size_actual, 1)  # [batch_size, model_embedding_dim]
                        batch_model_ids = torch.full(
                            (batch_size_actual,),
                            model_id_idx,
                            dtype=torch.long,
                            device=self.device,
                        )  # [batch_size]
                        
                        batch_predictions: torch.Tensor = self.network(
                            batch_embeddings,
                            batch_features,
                            batch_model_embedding,
                            batch_model_ids,
                        )  # [batch_size]
                        
                        # Add back per-model average, then convert scaled log-lengths to raw lengths
                        batch_predictions_np = batch_predictions.cpu().numpy()  # [batch_size]
                        batch_predictions_np = batch_predictions_np + self._model_avg_lengths.get(model_name, 0.0)  # [batch_size]
                        batch_predictions_log = self.scaler.inverse_transform(batch_predictions_np)  # [batch_size]
                        batch_predictions_raw = np.exp(batch_predictions_log)  # [batch_size]
                        model_predictions.append(batch_predictions_raw)
                    
                    predictions_dict[model_name] = np.concatenate(model_predictions) # [n_prompts]
            
            return LengthPredictionOutputData(predictions=predictions_dict)

    def get_history(self) -> TrainingHistory:
        """Get training history."""
        return TrainingHistory.from_entries(self._history_entries)

    def get_state_dict(self) -> dict[str, Any]:
        """
        Get state dictionary for saving the model.
        
        Returns:
            State dictionary containing all model parameters and configuration
        """
        if self._network is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        if not self.embedding_model.is_initialized:
            raise RuntimeError("Embedding model not initialized")
        
        if self._model_encoder is None:
            raise RuntimeError("Model encoder not initialized")

        return {
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "optimizer_state": self._optimizer_state,
            "scheduler_state": self._scheduler_state,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "input_proj_dim": self.input_proj_dim,
            "embedding_model_name": self.embedding_model_name,
            "print_every": self.print_every,
            "preprocessor_version": self.preprocessor.version,
            "prompt_embedding_dim": self._prompt_embedding_dim,
            "prompt_features_dim": self._prompt_features_dim,
            "n_models": self._n_models,
            "model_id_embedding_dim": self.model_id_embedding_dim,
            "model_encoder_state": self._model_encoder.get_state_dict(),
            "scaler_state": self.scaler.get_state_dict(),
            "prompt_features_scaler_state": self._prompt_features_scaler.get_state_dict(),
            "network_state_dict": state_dict_to_cpu(self.network.state_dict()),
            "epochs_completed": self._epochs_completed,
            "embedding_type": self.embedding_model.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump() if self.embedding_spec is not None else None,
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "embedding_model_state_dict": self.embedding_model.get_state_dict(),
            "seed": self.seed,
            "model_avg_lengths": self._model_avg_lengths,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any], instance: "DnEmbeddingLengthPredictionModel | None" = None) -> "DnEmbeddingLengthPredictionModel":
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
            embedding_spec = embedding_spec_adapter.validate_python(state_dict["embedding_spec"])
            
            model = cls(
                hidden_dims=state_dict["hidden_dims"],
                dropout=state_dict["dropout"],
                input_proj_dim=state_dict["input_proj_dim"],
                optimizer_spec=optimizer_spec,
                embedding_model_name=state_dict["embedding_model_name"],
                embedding_spec=embedding_spec,
                min_model_comparisons=state_dict["min_model_comparisons"],
                embedding_model_epochs=state_dict["embedding_model_epochs"],
                model_id_embedding_dim=state_dict["model_id_embedding_dim"],
                print_every=state_dict["print_every"],
                seed=state_dict["seed"],
            )
        
        model.embedding_model = EmbeddingModelBase.load_from_state_dict(state_dict["embedding_model_state_dict"])
        model._model_encoder = StringEncoder.load_state_dict(state_dict["model_encoder_state"])

        if model._network is None:
            model._initialize_network(
                prompt_embedding_dim=state_dict["prompt_embedding_dim"],
                prompt_features_dim=state_dict["prompt_features_dim"],
                n_models=state_dict["n_models"],
            )
        model.network.load_state_dict(state_dict["network_state_dict"])
        model.network.to(model.device)
        
        model._scaler = SimpleScaler.from_state_dict(state_dict["scaler_state"])
        model._prompt_features_scaler = SimpleScaler.from_state_dict(state_dict["prompt_features_scaler_state"])
        
        model._optimizer_state = state_dict.get("optimizer_state")
        model._scheduler_state = state_dict.get("scheduler_state")
        model._epochs_completed = state_dict.get("epochs_completed", 0)
        model._model_avg_lengths = state_dict.get("model_avg_lengths", {})
        
        return model

    def _compute_model_avg_lengths(
        self,
        preprocessed_data: PreprocessedLengthPredictionTrainingDataWithEmbeddings,
    ) -> dict[str, float]:
        """
        Compute per-model average scaled log-length from training data.

        The model will learn to predict residuals from these averages, encoding
        the per-model verbosity baseline separately from the prompt-driven deviation.

        Args:
            preprocessed_data: Preprocessed training data (training split only)

        Returns:
            Dict mapping model name to its mean scaled log-length across all training samples
        """
        model_lengths: dict[str, list[float]] = {}
        for sample in preprocessed_data.samples:
            name_a = preprocessed_data.model_encoder.decode(sample.model_id_a)
            name_b = preprocessed_data.model_encoder.decode(sample.model_id_b)
            model_lengths.setdefault(name_a, []).append(sample.log_response_length_a)
            model_lengths.setdefault(name_b, []).append(sample.log_response_length_b)
        return {name: float(np.mean(lengths)) for name, lengths in model_lengths.items()}

    def _prepare_dataloader(
        self, 
        preprocessed_data: PreprocessedLengthPredictionTrainingDataWithEmbeddings, 
        batch_size: int,
        model_avg_lengths: dict[str, float],
    ) -> _DataLoaderType:
        """
        Prepare dataloader from preprocessed training data.
        
        Args:
            preprocessed_data: Preprocessed training data
            batch_size: Batch size
            model_avg_lengths: Per-model average scaled log-length (training-split averages);
                               targets are stored as residuals relative to these averages
            
        Returns:
            DataLoader for training/validation
        """
        # Unpack pairs into individual samples
        prompt_embeddings_list = []  # [n_samples, embedding_dim]
        prompt_features_list = []  # [n_samples, prompt_features_dim]
        model_embeddings_list = []  # [n_samples, model_embedding_dim]
        model_ids_list = []  # [n_samples]
        lengths_list = []  # [n_samples]
        
        for sample in preprocessed_data.samples:
            name_a = preprocessed_data.model_encoder.decode(sample.model_id_a)
            name_b = preprocessed_data.model_encoder.decode(sample.model_id_b)
            avg_a = model_avg_lengths.get(name_a, 0.0)
            avg_b = model_avg_lengths.get(name_b, 0.0)

            # Add model A sample (target is residual from per-model average)
            prompt_embeddings_list.append(torch.from_numpy(sample.prompt_embedding))
            prompt_features_list.append(torch.from_numpy(sample.prompt_features))
            model_embeddings_list.append(torch.from_numpy(sample.model_embedding_a))
            model_ids_list.append(sample.model_id_a)
            lengths_list.append(sample.log_response_length_a - avg_a)
            
            # Add model B sample (target is residual from per-model average)
            prompt_embeddings_list.append(torch.from_numpy(sample.prompt_embedding))
            prompt_features_list.append(torch.from_numpy(sample.prompt_features))
            model_embeddings_list.append(torch.from_numpy(sample.model_embedding_b))
            model_ids_list.append(sample.model_id_b)
            lengths_list.append(sample.log_response_length_b - avg_b)
        
        prompt_embeddings = torch.stack(prompt_embeddings_list)  # [n_samples, embedding_dim]
        prompt_features = torch.stack(prompt_features_list)  # [n_samples, prompt_features_dim]
        model_embeddings = torch.stack(model_embeddings_list)  # [n_samples, model_embedding_dim]
        model_ids = torch.tensor(model_ids_list, dtype=torch.long)  # [n_samples]
        lengths = torch.tensor(lengths_list, dtype=torch.float32)  # [n_samples]
        
        dataset = TensorDataset(
            prompt_embeddings,
            prompt_features,
            model_embeddings,
            model_ids,
            lengths,
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
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

    def _train_epoch(
        self, 
        epoch: int,
        dataloader: _DataLoaderType,
        val_dataloader: _DataLoaderType | None,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epochs_timer: Timer,
    ) -> "DnEmbeddingLengthPredictionModel.EpochResult":
        with Timer(f"epoch_{epoch}", verbosity="start+end", parent=epochs_timer) as timer:
            self.network.train()
            total_loss = 0.0
            n_batches = 0
            total_samples = 0

            all_predictions = []
            all_actuals = []

            diagnostic_accumulator = EpochDiagnosticsAccumulator()
            trunk_params, prompt_emb_proj_params, prompt_feat_proj_params, model_emb_proj_params, model_id_emb_params = self.network.get_param_groups_for_metrics()
            numeric_descs, bool_descs = get_feature_descriptions()
            prompt_feature_names = [d.name for d in numeric_descs] + [d.name for d in bool_descs]

            for batch_emb, batch_features, batch_model_emb, batch_model_ids, batch_lengths in dataloader:
                batch_emb: torch.Tensor = batch_emb.to(self.device).requires_grad_(True)  # [batch_size, prompt_embedding_dim]
                batch_features: torch.Tensor = batch_features.to(self.device).requires_grad_(True)  # [batch_size, prompt_features_dim]
                batch_model_emb: torch.Tensor = batch_model_emb.to(self.device).requires_grad_(True)  # [batch_size, model_embedding_dim]
                batch_model_ids: torch.Tensor = batch_model_ids.to(self.device)  # [batch_size]
                batch_lengths: torch.Tensor = batch_lengths.to(self.device)  # [batch_size]

                total_samples += len(batch_emb)

                with torch.no_grad():
                    pe = self.network.prompt_emb_proj(batch_emb)        # [batch_size, input_proj_dim]
                    pf = self.network.prompt_feat_proj(batch_features)  # [batch_size, input_proj_dim]
                    me = self.network.model_emb_proj(batch_model_emb)   # [batch_size, input_proj_dim]
                diagnostic_accumulator.update_representation_stats({
                    "prompt_emb_proj": pe,
                    "prompt_feat_proj": pf,
                    "model_emb_proj": me,
                })

                optimizer.zero_grad()
                predictions: torch.Tensor = self.network(
                    batch_emb,
                    batch_features,
                    batch_model_emb,
                    batch_model_ids,
                )  # [batch_size]

                loss: torch.Tensor = criterion(predictions, batch_lengths)
                loss.backward()

                diagnostic_accumulator.update_grad_norms({
                    "trunk": trunk_params,
                    "prompt_emb_proj": prompt_emb_proj_params,
                    "prompt_feat_proj": prompt_feat_proj_params,
                    "model_emb_proj": model_emb_proj_params,
                    "model_id_embedding": model_id_emb_params,
                })
                diagnostic_accumulator.update_gradient_attribution({
                    **split_tensor_with_grad(batch_features, prompt_feature_names),
                    "prompt_embedding": batch_emb,
                    "model_embedding": batch_model_emb,
                })

                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                # Collect for metrics computation
                all_predictions.append(predictions.detach().cpu().numpy())
                all_actuals.append(batch_lengths.detach().cpu().numpy())
            
            avg_loss = total_loss / n_batches

            # Compute train metrics
            all_predictions_np = np.concatenate(all_predictions)
            all_actuals_np = np.concatenate(all_actuals)
            train_metrics = compute_length_prediction_metrics(all_predictions_np, all_actuals_np, self.scaler)

            with Timer("perform_validation", verbosity="start+end", parent=timer):
                val_loss, val_metrics = self._perform_validation(val_dataloader, criterion, timer) if val_dataloader is not None else (None, None)

            # Combine metrics
            additional_metrics: dict[str, float] = {
                "train_avg_relative_error": train_metrics["avg_relative_error"],
                "train_avg_relative_ratio": train_metrics["avg_relative_ratio"],
                "train_stddev_ratio": train_metrics["stddev_ratio"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                **diagnostic_accumulator.to_dict(),
            }
            if val_metrics is not None:
                additional_metrics.update({
                    "val_avg_relative_error": val_metrics["avg_relative_error"],
                    "val_avg_relative_ratio": val_metrics["avg_relative_ratio"],
                    "val_stddev_ratio": val_metrics["stddev_ratio"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                })
            
            entry = TrainingHistoryEntry(
                epoch=epoch,
                total_loss=avg_loss,
                val_loss=val_loss,
                train_accuracy=train_metrics["accuracy"],
                val_accuracy=val_metrics["accuracy"] if val_metrics is not None else None,
                additional_metrics=additional_metrics,
            )
            self._history_entries.append(entry)
            
            self.append_entry_to_log(entry, log_timings_from=self.last_timer)
            
        return self.EpochResult(
            epoch=epoch,
            total_loss=avg_loss,
            val_loss=val_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            duration=timer.elapsed_time,
        )

    def _perform_validation(
        self,
        val_dataloader: _DataLoaderType,
        criterion: nn.Module,
        timer: Timer,
    ) -> tuple[float, dict[str, float]]:
        self.network.eval()
        total_loss = 0.0
        n_batches = 0
        total_samples = 0
        
        all_predictions = []
        all_actuals = []
        
        for batch_emb, batch_features, batch_model_emb, batch_model_ids, batch_lengths in val_dataloader:
            with Timer(f"batch_{n_batches}", verbosity="start+end", parent=timer):
                batch_emb: torch.Tensor = batch_emb.to(self.device)  # [batch_size, embedding_dim]
                batch_features: torch.Tensor = batch_features.to(self.device)  # [batch_size, prompt_features_dim]
                batch_model_emb: torch.Tensor = batch_model_emb.to(self.device)  # [batch_size, model_embedding_dim]
                batch_model_ids: torch.Tensor = batch_model_ids.to(self.device)  # [batch_size]
                batch_lengths: torch.Tensor = batch_lengths.to(self.device)  # [batch_size]
                
                total_samples += len(batch_emb)
                
                with torch.no_grad():
                    predictions: torch.Tensor = self.network(
                        batch_emb,
                        batch_features,
                        batch_model_emb,
                        batch_model_ids,
                    )  # [batch_size]
                    
                    loss: torch.Tensor = criterion(predictions, batch_lengths)
                    
                    total_loss += loss.item()
                    n_batches += 1
                    
                    # Collect for metrics computation
                    all_predictions.append(predictions.cpu().numpy())
                    all_actuals.append(batch_lengths.cpu().numpy())
        
        avg_loss = total_loss / n_batches
        
        # Compute validation metrics
        all_predictions_np = np.concatenate(all_predictions)
        all_actuals_np = np.concatenate(all_actuals)
        val_metrics = compute_length_prediction_metrics(all_predictions_np, all_actuals_np, self.scaler)
        
        return avg_loss, val_metrics

    def _log_epoch_result(self, result: "DnEmbeddingLengthPredictionModel.EpochResult") -> None:
        if self.print_every is None:
            return
        
        if not result.epoch % self.print_every == 0:
            return
        
        train_metrics = result.train_metrics
        val_metrics = result.val_metrics
        
        if val_metrics is None:
            print(f"Epoch {result.epoch:>4}: loss = {result.total_loss:.4f}, "
                  f"accuracy = {(train_metrics['accuracy']*100):.2f}%, "
                  f"rel_err = {(train_metrics['avg_relative_error']*100):.2f}%, "
                  f"ratio = {train_metrics['avg_relative_ratio']:.3f}, "
                  f"mae = {train_metrics['mae']:.1f} - {result.duration:.2f}s")
        else:
            print(f"Epoch {result.epoch:>4}: loss = {result.total_loss:.4f}/{result.val_loss:.4f}, "
                  f"accuracy = {(train_metrics['accuracy']*100):.2f}%/{(val_metrics['accuracy']*100):.2f}%, "
                  f"rel_err = {(train_metrics['avg_relative_error']*100):.2f}%/{(val_metrics['avg_relative_error']*100):.2f}%, "
                  f"ratio = {train_metrics['avg_relative_ratio']:.3f}/{val_metrics['avg_relative_ratio']:.3f}, "
                  f"mae = {train_metrics['mae']:.1f}/{val_metrics['mae']:.1f} - {result.duration:.2f}s")

    @dataclass
    class EpochResult:
        epoch: int
        total_loss: float
        val_loss: float | None
        train_metrics: dict[str, float]
        val_metrics: dict[str, float] | None
        duration: float

    class _ResidualBlock(nn.Module):
        """Dense layer with a linear skip connection (projection shortcut).

        The main path applies Linear → LeakyReLU → Dropout; the shortcut is a
        bias-free linear projection that matches dimensions.  The two paths are
        summed so that gradients can flow directly through the shortcut, easing
        optimisation of deeper networks.
        """

        def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
            super().__init__()
            self.main = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(dropout),
            )
            self.shortcut = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # [batch_size, in_dim] -> [batch_size, out_dim]
            return self.main(x) + self.shortcut(x)

    class _DenseNetwork(nn.Module):
        def __init__(
            self,
            prompt_embedding_dim: int,
            prompt_features_dim: int,
            model_embedding_dim: int,
            n_models: int,
            model_id_embedding_dim: int,
            input_proj_dim: int,
            hidden_dims: list[int],
            dropout: float = 0.2,
        ) -> None:
            super().__init__()

            self.model_id_embedding = nn.Embedding(
                num_embeddings=n_models,
                embedding_dim=model_id_embedding_dim,
            )

            self.prompt_emb_proj = nn.Sequential(nn.Linear(prompt_embedding_dim, input_proj_dim), nn.LeakyReLU(0.1))
            self.prompt_feat_proj = nn.Sequential(nn.Linear(prompt_features_dim, input_proj_dim), nn.LeakyReLU(0.1))
            self.model_emb_proj = nn.Sequential(nn.Linear(model_embedding_dim, input_proj_dim), nn.LeakyReLU(0.1))

            blocks = []
            prev_dim = input_proj_dim * 3 + model_id_embedding_dim

            for hidden_dim in hidden_dims:
                blocks.append(DnEmbeddingLengthPredictionModel._ResidualBlock(prev_dim, hidden_dim, dropout))
                prev_dim = hidden_dim

            self.blocks = nn.ModuleList(blocks)
            self.output_layer = nn.Linear(prev_dim, 1)

            self._init_weights()

        def forward(
            self,
            prompt_embedding: torch.Tensor,  # [batch_size, prompt_embedding_dim]
            prompt_features: torch.Tensor,  # [batch_size, prompt_features_dim]
            model_embedding: torch.Tensor,  # [batch_size, model_embedding_dim]
            model_id: torch.Tensor,  # [batch_size] -- integer IDs; -1 means unknown (uses mean of all embeddings)
        ) -> torch.Tensor:
            known_mask = model_id >= 0  # [batch_size]
            # Clamp to valid range so the lookup never errors; unknown rows are overwritten below
            id_embs = self.model_id_embedding(model_id.clamp(min=0))  # [batch_size, model_id_embedding_dim]
            mean_emb = self.model_id_embedding.weight.mean(dim=0)  # [model_id_embedding_dim]
            id_embs = torch.where(known_mask.unsqueeze(-1), id_embs, mean_emb)  # [batch_size, model_id_embedding_dim]

            pe = self.prompt_emb_proj(prompt_embedding)   # [batch_size, input_proj_dim]
            pf = self.prompt_feat_proj(prompt_features)   # [batch_size, input_proj_dim]
            me = self.model_emb_proj(model_embedding)     # [batch_size, input_proj_dim]

            x = torch.cat([pe, pf, me, id_embs], dim=1)  # [batch_size, input_proj_dim * 3 + model_id_embedding_dim]
            for block in self.blocks:
                x = block(x)  # [batch_size, hidden_dim]
            output: torch.Tensor = self.output_layer(x)  # [batch_size, 1]
            return output.squeeze(-1)  # [batch_size]

        def get_param_groups_for_metrics(
            self,
        ) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
            """Return (trunk_params, prompt_emb_proj_params, prompt_feat_proj_params, model_emb_proj_params, model_id_embedding_params) for gradient norm metrics."""
            trunk_params = list(self.blocks.parameters()) + list(self.output_layer.parameters())
            return (
                trunk_params,
                list(self.prompt_emb_proj.parameters()),
                list(self.prompt_feat_proj.parameters()),
                list(self.model_emb_proj.parameters()),
                list(self.model_id_embedding.parameters()),
            )

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
