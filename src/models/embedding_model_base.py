"""Base class for model behavior embedding models."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from src.data_models.data_models import TrainingData
from src.utils.data_split import ValidationSplit


class EmbeddingModelBase(ABC):
    """
    Base class for model behavior embedding models.
    
    These models learn to map (prompt, response) pairs to an embedding space where
    models with similar characteristics are close together.
    
    This is used as a component within other models (e.g., DnEmbeddingModel) to
    learn model representations that can be used for routing or other tasks.
    """
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get the dimensionality of the output embeddings."""
        pass
    
    @property
    @abstractmethod
    def model_embeddings(self) -> dict[str, np.ndarray]:
        """Get the computed model embeddings (must be initialized after training)."""
        pass
    
    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the model has been initialized (trained or loaded)."""
        pass
    
    @abstractmethod
    def train(
        self,
        data: TrainingData,
        validation_split: ValidationSplit | None = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        """
        Train the embedding model on the given data.
        
        Args:
            data: Training data
            validation_split: Configuration for train/val split (if None, no validation)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        pass
    
    @abstractmethod
    def get_state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization."""
        pass
    
    @staticmethod
    def load_from_state_dict(state_dict: dict[str, Any]) -> "EmbeddingModelBase":
        """
        Load embedding model from state dict based on model type.
        
        This is a factory method that dispatches to the appropriate model class.
        
        Args:
            state_dict: State dictionary containing model_type and model parameters
            
        Returns:
            Loaded embedding model instance
        """
        model_type = state_dict.get("model_type")
        
        if model_type == "TripletFrozenEncoderModel":
            from src.models.triplet_frozen_encoder_model import TripletFrozenEncoderModel
            return TripletFrozenEncoderModel.load_state_dict(state_dict)
        elif model_type == "TripletFinetunableEncoderModel":
            from src.models.triplet_finetunable_encoder_model import TripletFinetunableEncoderModel
            return TripletFinetunableEncoderModel.load_state_dict(state_dict)
        elif model_type == "AttentionEmbeddingModel":
            from src.models.attention_embedding_model import AttentionEmbeddingModel
            return AttentionEmbeddingModel.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unknown embedding model type in state: {model_type}")

