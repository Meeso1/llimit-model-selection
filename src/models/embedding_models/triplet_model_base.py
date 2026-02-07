"""Base class for triplet-based model behavior encoders."""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Generic
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_models.data_models import TrainingData
from src.data_models.triplet_encoder_types import (
    PreprocessedTripletEncoderData,
    PromptResponsePair,
    TripletType,
)
from src.models.embedding_models.embedding_model_base import EmbeddingModelBase
from src.utils.data_split import ValidationSplit
from src.utils.timer import Timer


class TripletModelBase(EmbeddingModelBase, ABC, Generic[TripletType]):
    """
    Base class for triplet-based model behavior encoders.
    
    This encoder learns to map (prompt, response) pairs to an embedding space where:
    - Models with similar performance are close together
    - Each model has a distinctive signature
    - Winning responses are separated from losing responses
    
    Subclasses must implement the specific encoder architecture (frozen vs fine-tunable).
    """
    
    def __init__(
        self,
        triplet_margin: float = 0.2,
        regularization_weight: float = 0.01,
        min_model_comparisons: int = 20,
        identity_positive_ratio: float = 0.8,
        preprocessor_seed: int = 42,
        print_every: int | None = None,
    ) -> None:
        """
        Initialize the triplet-based encoder.
        
        Args:
            triplet_margin: Margin for triplet loss
            regularization_weight: Weight for KL-divergence regularization loss
            min_model_comparisons: Minimum comparisons for a model to be included
            identity_positive_ratio: Ratio of identity vs performance positives
            preprocessor_seed: Random seed for preprocessor
            print_every: Print progress every N epochs (None = no printing)
        """
        self.triplet_margin = triplet_margin
        self.regularization_weight = regularization_weight
        self.min_model_comparisons = min_model_comparisons
        self.identity_positive_ratio = identity_positive_ratio
        self.preprocessor_seed = preprocessor_seed
        self.print_every = print_every
        
        self._model_embeddings: dict[str, np.ndarray] | None = None
        self._epoch_logs: list["TripletModelBase.EpochLog"] = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_timer: Timer | None = None
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get the dimensionality of the output embeddings."""
        pass
    
    @property
    def model_embeddings(self) -> dict[str, np.ndarray]:
        """Get the model embeddings (must be initialized first)."""
        if self._model_embeddings is None:
            raise RuntimeError("Model embeddings not initialized. Train or load a model first.")
        return self._model_embeddings
    
    @property
    def is_initialized(self) -> bool:
        """Check if the model has been initialized (trained or loaded)."""
        return self._get_module() is not None
    
    @abstractmethod
    def _initialize_module(self, input_dim: int | None = None) -> None:
        """Initialize the neural network module."""
        pass

    @abstractmethod
    def _infer_input_dim(self, first_batch: Any) -> int | None:
        """Infer the input dimension from the first batch."""
        pass
    
    @abstractmethod
    def _get_module(self) -> nn.Module:
        """Get the neural network module."""
        pass
    
    @abstractmethod
    def _prepare_dataloader(
        self,
        preprocessed_data: PreprocessedTripletEncoderData[TripletType],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Prepare dataloader from preprocessed data."""
        pass
    
    @abstractmethod
    def _preprocess_data(self, data: TrainingData) -> PreprocessedTripletEncoderData[TripletType]:
        """Preprocess training data."""
        pass
    
    @abstractmethod
    def _split_preprocessed_data(
        self,
        preprocessed_data: PreprocessedTripletEncoderData[TripletType],
        val_fraction: float,
        seed: int,
    ) -> tuple[PreprocessedTripletEncoderData[TripletType], PreprocessedTripletEncoderData[TripletType]]:
        """Split preprocessed data into train and validation sets."""
        pass
    
    @abstractmethod
    def _train_epoch(
        self,
        epoch: int,
        dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epochs_timer: Timer,
    ) -> "TripletModelBase.EpochLog":
        """Train for one epoch."""
        pass
    
    def train(
        self,
        data: TrainingData,
        validation_split: ValidationSplit | None = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        """
        Train the model on the given data.
        
        Args:
            data: Training data (will be split if validation_split is provided)
            validation_split: Configuration for train/val split (if None, no validation)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        with Timer("train_triplet_encoder", verbosity="start+end") as train_timer:
            self.last_timer = train_timer
            
            with Timer("preprocess", verbosity="start+end", parent=train_timer):
                preprocessed_data = self._preprocess_data(data)
            
            with Timer("split_data", verbosity="start+end", parent=train_timer):
                if validation_split is not None:
                    train_preprocessed, val_preprocessed = self._split_preprocessed_data(
                        preprocessed_data,
                        val_fraction=validation_split.val_fraction,
                        seed=validation_split.seed,
                    )
                else:
                    train_preprocessed = preprocessed_data
                    val_preprocessed = None
            
            with Timer("prepare_dataloaders", verbosity="start+end", parent=train_timer):
                train_dataloader = self._prepare_dataloader(train_preprocessed, batch_size, shuffle=True)
                val_dataloader = self._prepare_dataloader(val_preprocessed, batch_size, shuffle=False) \
                    if val_preprocessed is not None else None
            
            if self._get_module() is None:
                # Get first batch so that input dimension can be inferred
                first_batch = next(iter(train_dataloader))
                self._initialize_module(self._infer_input_dim(first_batch))
            
            optimizer = self._create_optimizer()
            scheduler = self._create_scheduler(optimizer)
            
            criterion = nn.TripletMarginLoss(margin=self.triplet_margin, p=2)
            
            with Timer("epochs", verbosity="start+end", parent=train_timer) as epochs_timer:
                for epoch in range(1, epochs + 1):
                    epoch_log = self._train_epoch(
                        epoch,
                        train_dataloader,
                        val_dataloader,
                        optimizer,
                        criterion,
                        epochs_timer,
                    )
                    
                    self._log_epoch_result(epoch_log)
                    
                    if scheduler is not None:
                        scheduler.step()
            
            with Timer("compute_model_embeddings", verbosity="start+end", parent=train_timer):
                self._model_embeddings = self._compute_model_embeddings(data)
    
    def _compute_model_embeddings_from_stored(
        self,
        anchor_embeddings: torch.Tensor,  # [n_samples, embedding_dim]
        anchor_model_ids: list[str],  # [n_samples]
    ) -> dict[str, np.ndarray]:
        """
        Compute model embeddings by averaging stored embeddings from each model.
        
        Args:
            anchor_embeddings: Anchor embeddings from training/validation batch
            anchor_model_ids: Model IDs for each anchor
            
        Returns:
            Dictionary mapping model IDs to their average embeddings
        """
        if len(anchor_embeddings) == 0:
            raise ValueError("anchor_embeddings cannot be empty")
        
        # Group embeddings by model
        model_embeddings_dict: dict[str, list[torch.Tensor]] = defaultdict(list)
        for emb, model_id in zip(anchor_embeddings, anchor_model_ids):
            model_embeddings_dict[model_id].append(emb)
        
        # Average embeddings for each model and convert to numpy
        model_embeddings = {
            model_id: torch.stack(embs).mean(dim=0).detach().cpu().numpy()
            for model_id, embs in model_embeddings_dict.items()
        }
        
        return model_embeddings
    
    @abstractmethod
    def encode(
        self,
        pairs: list[PromptResponsePair],
    ) -> np.ndarray:
        """
        Encode a list of (prompt, response) pairs into embeddings.
        
        Args:
            pairs: List of prompt-response pairs  # [n_samples]
            
        Returns:
            Array of embeddings  # [n_samples, embedding_dim]
        """
        pass
    
    def compute_model_embedding(
        self,
        pairs: list[PromptResponsePair],
    ) -> np.ndarray:
        """
        Compute a single, representative embedding for a model by averaging.
        
        Args:
            pairs: List of prompt-response pairs from the model  # [n_samples]
            
        Returns:
            Single averaged embedding  # [embedding_dim]
        """
        embeddings = self.encode(pairs)  # [n_samples, embedding_dim]
        return np.mean(embeddings, axis=0)  # [embedding_dim]
    
    @abstractmethod
    def get_state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization."""
        pass
    
    @abstractmethod
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for the model."""
        pass
    
    @abstractmethod
    def _create_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler | None:
        """Create learning rate scheduler."""
        pass
    
    def _compute_regularization_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence regularization loss.
        
        Encourages embeddings to follow a standard normal distribution.
        
        Args:
            embeddings: Batch of embeddings  # [batch_size, embedding_dim]
            
        Returns:
            Scalar regularization loss
        """
        # Compute mean and log variance across the batch
        mean = embeddings.mean(dim=0)  # [embedding_dim]
        log_var = torch.log(embeddings.var(dim=0) + 1e-8)  # [embedding_dim]
        
        # KL divergence from N(0, 1)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return kl_loss / embeddings.size(0)  # Normalize by batch size
    
    def _compute_model_embeddings(self, data: TrainingData) -> dict[str, np.ndarray]:
        """Compute model embeddings for the given data."""
        all_prompts = [
            v for entry in data.entries for v in [
                (entry.model_a, PromptResponsePair(prompt=entry.user_prompt, response=entry.model_a_response)),
                (entry.model_b, PromptResponsePair(prompt=entry.user_prompt, response=entry.model_b_response)),
            ]
        ]
        prompts_by_model: dict[str, list[PromptResponsePair]] = defaultdict(list)
        for model, prompt_response_pair in all_prompts:
            prompts_by_model[model].append(prompt_response_pair)
        
        embeddings = {
            model: self.compute_model_embedding(prompts)
            for model, prompts in prompts_by_model.items()
        }
        
        default_embedding = self.compute_model_embedding([v for _, v in all_prompts])
        embeddings["default"] = default_embedding
        return embeddings
    
    def _log_epoch_result(self, epoch_log: "TripletModelBase.EpochLog") -> None:
        """Print epoch results if print_every is set."""
        if self.print_every is None:
            return
        
        if epoch_log.epoch % self.print_every != 0:
            return
        
        accuracy_str = f"triplet_acc = {(epoch_log.train_triplet_accuracy*100):.2f}%"
        if epoch_log.val_triplet_accuracy is not None:
            accuracy_str += f"/{(epoch_log.val_triplet_accuracy*100):.2f}%"
        
        univ_acc_str = f"univ_acc = {(epoch_log.train_universal_accuracy*100):.2f}%"
        if epoch_log.val_universal_accuracy is not None:
            univ_acc_str += f"/{(epoch_log.val_universal_accuracy*100):.2f}%"
            
        if epoch_log.val_loss is None:
            print(
                f"Epoch {epoch_log.epoch:>4}: "
                f"loss = {epoch_log.train_loss:.4f} "
                f"(triplet: {epoch_log.train_triplet_loss:.4f}, reg: {epoch_log.train_reg_loss:.4f}), "
                f"{accuracy_str}, {univ_acc_str} - "
                f"{epoch_log.duration:.2f}s"
            )
        else:
            print(
                f"Epoch {epoch_log.epoch:>4}: "
                f"loss = {epoch_log.train_loss:.4f}/{epoch_log.val_loss:.4f}, "
                f"{accuracy_str}, {univ_acc_str} - "
                f"{epoch_log.duration:.2f}s"
            )
    
    @dataclass
    class EpochLog:
        """Log entry for a single training epoch."""
        epoch: int
        train_loss: float
        train_triplet_loss: float
        train_reg_loss: float
        train_triplet_accuracy: float
        train_universal_accuracy: float
        val_loss: float | None
        val_triplet_accuracy: float | None
        val_universal_accuracy: float | None
        duration: float

