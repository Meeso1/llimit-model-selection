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
from src.utils.wandb_details import WandbDetails
from src.utils.timer import Timer
from src.utils.torch_utils import state_dict_to_cpu
from src.utils.data_split import ValidationSplit, split_length_prediction_preprocessed_data
from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models import model_loading


_DataLoaderType = DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


class DnEmbeddingLengthPredictionModel(LengthPredictionModelBase):
    """Dense network model for predicting response lengths."""
    
    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        optimizer_spec: OptimizerSpecification | None = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_spec: EmbeddingSpec | None = None,
        load_embedding_model_from: str | None = None,
        min_model_comparisons: int = 20,
        embedding_model_epochs: int = 10,
        wandb_details: WandbDetails | None = None,
        print_every: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(wandb_details)

        if load_embedding_model_from is None and embedding_spec is None:
            raise ValueError("Either embedding_spec or load_embedding_model_from must be specified")
        
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 128, 64]
        self.optimizer_spec = optimizer_spec if optimizer_spec is not None else AdamWSpec(learning_rate=0.001)
        self.embedding_model_name = embedding_model_name
        self.print_every = print_every
        self.min_model_comparisons = min_model_comparisons
        self.embedding_model_epochs = embedding_model_epochs
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
    ) -> None:
        self._prompt_embedding_dim = prompt_embedding_dim
        self._prompt_features_dim = prompt_features_dim
        self._network = self._DenseNetwork(
            prompt_embedding_dim=prompt_embedding_dim,
            prompt_features_dim=prompt_features_dim,
            model_embedding_dim=self.embedding_model.embedding_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

    def get_config_for_wandb(self) -> dict[str, Any]:
        """Get configuration dictionary for Weights & Biases logging."""
        return {
            "model_type": "dn_embedding_length_prediction",
            "hidden_dims": self.hidden_dims,
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "embedding_model_name": self.embedding_model_name,
            "preprocessor_version": self.preprocessor.version,
            "embedding_type": self.embedding_spec.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump(),
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
            
            with Timer("encode_prompts", verbosity="start+end", parent=train_timer):
                preprocessed_without_embeddings = self.preprocessor.preprocess(data)
            
            with Timer("add_model_embeddings", verbosity="start+end", parent=train_timer):
                preprocessed_data = preprocessed_without_embeddings.add_model_embeddings(
                    self.model_embeddings,
                    self.embedding_model.embedding_dim,
                )
            
            self._scaler = SimpleScaler()
            self._scaler.load_state_dict(preprocessed_data.scaler_state)
                
            if self._network is None:
                self._initialize_network(
                    prompt_embedding_dim=preprocessed_data.embedding_dim,
                    prompt_features_dim=preprocessed_data.prompt_features_dim,
                )
            
            with Timer("split_preprocessed_data", verbosity="start+end", parent=train_timer):
                preprocessed_train, preprocessed_val = split_length_prediction_preprocessed_data(
                    preprocessed_data,
                    val_fraction=validation_split.val_fraction if validation_split is not None else 0,
                    seed=validation_split.seed if validation_split is not None else 42,
                )
            
            with Timer("prepare_dataloaders", verbosity="start+end", parent=train_timer):
                dataloader = self._prepare_dataloader(preprocessed_train, batch_size)
                val_dataloader = self._prepare_dataloader(preprocessed_val, batch_size) if preprocessed_val is not None else None
            
            optimizer, scheduler = self._create_optimizer_and_scheduler()
            
            # Use MSE loss for regression
            criterion = nn.MSELoss()
            
            with Timer("epochs", verbosity="start+end", parent=train_timer) as epochs_timer:
                for epoch in range(self._epochs_completed + 1, self._epochs_completed + epochs + 1):
                    result = self._train_epoch(epoch, dataloader, val_dataloader, optimizer, criterion, epochs_timer)
                    
                    self._log_epoch_result(result)
                    
                    if scheduler is not None:
                        scheduler.step()
                    
                    self._epochs_completed = epoch
            
            self._optimizer_state = optimizer.state_dict()
            self._scheduler_state = scheduler.state_dict() if scheduler is not None else None
            
            self.finish_wandb_if_needed()

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
        if self._network is None:
            raise RuntimeError("Model not trained or loaded yet")
        
        with Timer("predict", verbosity="start+end") as predict_timer:
            self.last_timer = predict_timer
            with Timer("preprocess_input", verbosity="start+end", parent=predict_timer):
                encoded_prompts = self.preprocessor.preprocess_for_inference(
                    prompts=X.prompts,
                    model_names=X.model_names,
                    model_encoder=StringEncoder()  # We don't need model IDs here
                )
            
            prompt_embeddings = torch.from_numpy(encoded_prompts.prompt_embeddings).to(self.device)  # [n_prompts, embedding_dim]
            prompt_features = torch.from_numpy(encoded_prompts.prompt_features).to(self.device)  # [n_prompts, prompt_features_dim]
            
            self.network.eval()
            predictions_dict: dict[str, np.ndarray] = {}
            
            with torch.no_grad():
                for model_name in X.model_names:
                    if model_name not in self.model_embeddings:
                        # Use default embedding if model not seen during training
                        model_embedding = self.model_embeddings.get("default", np.zeros(self.embedding_model.embedding_dim))
                    else:
                        model_embedding = self.model_embeddings[model_name]
                    
                    model_embedding_tensor = torch.from_numpy(model_embedding).to(self.device)
                    model_predictions = []
                    
                    for i in range(0, len(prompt_embeddings), batch_size):
                        batch_embeddings = prompt_embeddings[i:i + batch_size]  # [batch_size, embedding_dim]
                        batch_features = prompt_features[i:i + batch_size]  # [batch_size, prompt_features_dim]
                        batch_size_actual = len(batch_embeddings)
                        
                        # Repeat model embedding for batch
                        batch_model_embedding = model_embedding_tensor.unsqueeze(0).repeat(batch_size_actual, 1)  # [batch_size, model_embedding_dim]
                        
                        batch_predictions: torch.Tensor = self.network(
                            batch_embeddings,
                            batch_features,
                            batch_model_embedding,
                        )  # [batch_size]
                        
                        # Inverse transform using scaler to get back to original scale
                        batch_predictions_np = batch_predictions.cpu().numpy()  # [batch_size]
                        batch_predictions_descaled = self.scaler.transform(batch_predictions_np)  # [batch_size]
                        model_predictions.append(batch_predictions_descaled)
                    
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
        
        return {
            "optimizer_type": self.optimizer_spec.optimizer_type,
            "optimizer_params": self.optimizer_spec.to_dict(),
            "optimizer_state": self._optimizer_state,
            "scheduler_state": self._scheduler_state,
            "hidden_dims": self.hidden_dims,
            "embedding_model_name": self.embedding_model_name,
            "print_every": self.print_every,
            "preprocessor_version": self.preprocessor.version,
            "prompt_embedding_dim": self._prompt_embedding_dim,
            "prompt_features_dim": self._prompt_features_dim,
            "scaler_state": self.scaler.get_state_dict(),
            "network_state_dict": state_dict_to_cpu(self.network.state_dict()),
            "history_entries": self._history_entries,
            "epochs_completed": self._epochs_completed,
            "embedding_type": self.embedding_spec.embedding_type,
            "embedding_spec": self.embedding_spec.model_dump(),
            "min_model_comparisons": self.min_model_comparisons,
            "embedding_model_epochs": self.embedding_model_epochs,
            "embedding_model_state_dict": self.embedding_model.get_state_dict(),
            "seed": self.seed,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "DnEmbeddingLengthPredictionModel":
        """
        Load model from state dictionary.
        
        Args:
            state_dict: State dictionary from get_state_dict()
            
        Returns:
            Loaded model instance
        """
        optimizer_spec = OptimizerSpecification.from_serialized(
            state_dict["optimizer_type"],
            state_dict["optimizer_params"],
        )
        
        # Parse embedding spec using Pydantic TypeAdapter
        embedding_spec_adapter = TypeAdapter(EmbeddingSpec)
        embedding_spec = embedding_spec_adapter.validate_python(state_dict["embedding_spec"])
        
        model = cls(
            hidden_dims=state_dict["hidden_dims"],
            optimizer_spec=optimizer_spec,
            embedding_model_name=state_dict["embedding_model_name"],
            embedding_spec=embedding_spec,
            min_model_comparisons=state_dict["min_model_comparisons"],
            embedding_model_epochs=state_dict["embedding_model_epochs"],
            print_every=state_dict["print_every"],
            seed=state_dict["seed"],
        )
        
        model.embedding_model = EmbeddingModelBase.load_from_state_dict(state_dict["embedding_model_state_dict"])

        model._initialize_network(
            prompt_embedding_dim=state_dict["prompt_embedding_dim"],
            prompt_features_dim=state_dict["prompt_features_dim"],
        )
        model.network.load_state_dict(
            state_dict["network_state_dict"], 
        )
        model.network.to(model.device)
        
        model._scaler = SimpleScaler()
        model._scaler.load_state_dict(state_dict["scaler_state"])
        
        model._history_entries = state_dict["history_entries"]
        model._optimizer_state = state_dict.get("optimizer_state")
        model._scheduler_state = state_dict.get("scheduler_state")
        model._epochs_completed = state_dict.get("epochs_completed", 0)
        
        return model

    def _prepare_dataloader(
        self, 
        preprocessed_data: PreprocessedLengthPredictionTrainingDataWithEmbeddings, 
        batch_size: int,
    ) -> _DataLoaderType:
        """
        Prepare dataloader from preprocessed training data.
        
        Args:
            preprocessed_data: Preprocessed training data
            batch_size: Batch size
            
        Returns:
            DataLoader for training/validation
        """
        # Unpack pairs into individual samples
        prompt_embeddings_list = []  # [n_samples, embedding_dim]
        prompt_features_list = []  # [n_samples, prompt_features_dim]
        model_embeddings_list = []  # [n_samples, model_embedding_dim]
        lengths_list = []  # [n_samples]
        
        for sample in preprocessed_data.samples:
            # Add model A sample
            prompt_embeddings_list.append(torch.from_numpy(sample.prompt_embedding))
            prompt_features_list.append(torch.from_numpy(sample.prompt_features))
            model_embeddings_list.append(torch.from_numpy(sample.model_embedding_a))
            lengths_list.append(sample.response_length_a)
            
            # Add model B sample
            prompt_embeddings_list.append(torch.from_numpy(sample.prompt_embedding))
            prompt_features_list.append(torch.from_numpy(sample.prompt_features))
            model_embeddings_list.append(torch.from_numpy(sample.model_embedding_b))
            lengths_list.append(sample.response_length_b)
        
        prompt_embeddings = torch.stack(prompt_embeddings_list)  # [n_samples, embedding_dim]
        prompt_features = torch.stack(prompt_features_list)  # [n_samples, prompt_features_dim]
        model_embeddings = torch.stack(model_embeddings_list)  # [n_samples, model_embedding_dim]
        lengths = torch.tensor(lengths_list, dtype=torch.float32)  # [n_samples]
        
        dataset = TensorDataset(
            prompt_embeddings,
            prompt_features,
            model_embeddings,
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
            
            for batch_emb, batch_features, batch_model_emb, batch_lengths in dataloader:
                batch_emb: torch.Tensor = batch_emb.to(self.device)  # [batch_size, prompt_embedding_dim]
                batch_features: torch.Tensor = batch_features.to(self.device)  # [batch_size, prompt_features_dim]
                batch_model_emb: torch.Tensor = batch_model_emb.to(self.device)  # [batch_size, model_embedding_dim]
                batch_lengths: torch.Tensor = batch_lengths.to(self.device)  # [batch_size]
                
                total_samples += len(batch_emb)
                
                optimizer.zero_grad()
                predictions: torch.Tensor = self.network(
                    batch_emb,
                    batch_features,
                    batch_model_emb,
                )  # [batch_size]
                
                loss: torch.Tensor = criterion(predictions, batch_lengths)
                loss.backward()
                
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
            train_metrics = self._compute_metrics(all_predictions_np, all_actuals_np)
            
            with Timer("perform_validation", verbosity="start+end", parent=timer):
                val_loss, val_metrics = self._perform_validation(val_dataloader, criterion, timer) if val_dataloader is not None else (None, None)
            
            # Combine metrics
            additional_metrics = {
                "train_avg_relative_error": train_metrics["avg_relative_error"],
                "train_avg_relative_ratio": train_metrics["avg_relative_ratio"],
                "train_stddev_ratio": train_metrics["stddev_ratio"],
                "train_mae": train_metrics["mae"],
            }
            if val_metrics is not None:
                additional_metrics.update({
                    "val_avg_relative_error": val_metrics["avg_relative_error"],
                    "val_avg_relative_ratio": val_metrics["avg_relative_ratio"],
                    "val_stddev_ratio": val_metrics["stddev_ratio"],
                    "val_mae": val_metrics["mae"],
                })
            
            entry = TrainingHistoryEntry(
                epoch=epoch,
                total_loss=avg_loss,
                val_loss=val_loss,
                train_accuracy=train_metrics["avg_relative_error"],
                val_accuracy=val_metrics["avg_relative_error"] if val_metrics is not None else None,
                additional_metrics=additional_metrics,
            )
            self._history_entries.append(entry)
            
            if self.wandb_details is not None:
                self.log_to_wandb(entry)
            
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
        
        for batch_emb, batch_features, batch_model_emb, batch_lengths in val_dataloader:
            with Timer(f"batch_{n_batches}", verbosity="start+end", parent=timer):
                batch_emb: torch.Tensor = batch_emb.to(self.device)  # [batch_size, embedding_dim]
                batch_features: torch.Tensor = batch_features.to(self.device)  # [batch_size, prompt_features_dim]
                batch_model_emb: torch.Tensor = batch_model_emb.to(self.device)  # [batch_size, model_embedding_dim]
                batch_lengths: torch.Tensor = batch_lengths.to(self.device)  # [batch_size]
                
                total_samples += len(batch_emb)
                
                with torch.no_grad():
                    predictions: torch.Tensor = self.network(
                        batch_emb,
                        batch_features,
                        batch_model_emb,
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
        val_metrics = self._compute_metrics(all_predictions_np, all_actuals_np)
        
        return avg_loss, val_metrics

    def _compute_metrics(
        self,
        predictions: np.ndarray,  # [n_samples] - scaled
        actuals: np.ndarray,  # [n_samples] - scaled
    ) -> dict[str, float]:
        """
        Compute evaluation metrics for length prediction.
        
        Metrics computed:
        - avg_relative_error: 1 - abs(1 - predicted/actual) (higher is better, max 1.0)
        - avg_relative_ratio: mean(predicted/actual) (should be close to 1.0)
        - stddev_ratio: stddev(predictions)/stddev(actuals) (should be close to 1.0)
        - mae: mean absolute error in descaled space
        
        Args:
            predictions: Predicted lengths (scaled)
            actuals: Actual lengths (scaled)
            
        Returns:
            Dictionary of metrics
        """
        predictions_descaled = self.scaler.transform(predictions)  # [n_samples]
        actuals_descaled = self.scaler.transform(actuals)  # [n_samples]
        
        # Avoid division by zero
        actuals_nonzero = np.where(actuals_descaled == 0, 1e-6, actuals_descaled)
        
        # Average relative error magnitude: 1 - abs(1 - predicted/actual)
        relative_ratios = predictions_descaled / actuals_nonzero
        avg_relative_error = float(1 - np.mean(np.abs(1 - relative_ratios)))
        
        # Average relative ratio: mean(predicted/actual)
        avg_relative_ratio = float(np.mean(relative_ratios))
        
        # Stddev ratio: stddev(predictions)/stddev(actuals)
        pred_std = np.std(predictions_descaled)
        actual_std = np.std(actuals_descaled)
        stddev_ratio = float(pred_std / actual_std) if actual_std > 0 else 1.0
        
        # Mean absolute error in descaled space
        mae = float(np.mean(np.abs(predictions_descaled - actuals_descaled)))
        
        return {
            "avg_relative_error": avg_relative_error,
            "avg_relative_ratio": avg_relative_ratio,
            "stddev_ratio": stddev_ratio,
            "mae": mae,
        }

    def _log_epoch_result(self, result: "DnEmbeddingLengthPredictionModel.EpochResult") -> None:
        if self.print_every is None:
            return
        
        if not result.epoch % self.print_every == 0:
            return
        
        train_metrics = result.train_metrics
        val_metrics = result.val_metrics
        
        if val_metrics is None:
            print(f"Epoch {result.epoch:>4}: loss = {result.total_loss:.4f}, "
                  f"rel_err = {(train_metrics['avg_relative_error']*100):.2f}%, "
                  f"ratio = {train_metrics['avg_relative_ratio']:.3f}, "
                  f"mae = {train_metrics['mae']:.1f} - {result.duration:.2f}s")
        else:
            print(f"Epoch {result.epoch:>4}: loss = {result.total_loss:.4f}/{result.val_loss:.4f}, "
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

    class _DenseNetwork(nn.Module):
        def __init__(
            self,
            prompt_embedding_dim: int,
            prompt_features_dim: int,
            model_embedding_dim: int,
            hidden_dims: list[int],
        ) -> None:
            super().__init__()
            
            layers = []
            prev_dim = prompt_embedding_dim + prompt_features_dim + model_embedding_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LeakyReLU(negative_slope=0.01))
                layers.append(nn.Dropout(0.2))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            
            self.dense_network = nn.Sequential(*layers)

        def forward(
            self,
            prompt_embedding: torch.Tensor,  # [batch_size, prompt_embedding_dim]
            prompt_features: torch.Tensor,  # [batch_size, prompt_features_dim]
            model_embedding: torch.Tensor,  # [batch_size, model_embedding_dim]
        ) -> torch.Tensor:
            combined = torch.cat([prompt_embedding, prompt_features, model_embedding], dim=1)  # [batch_size, prompt_embedding_dim + prompt_features_dim + model_embedding_dim]
            output: torch.Tensor = self.dense_network(combined)  # [batch_size, 1]
            
            return output.squeeze(-1)  # [batch_size]
