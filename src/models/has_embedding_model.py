"""Protocol for models that contain embedding models."""

from typing import Protocol, runtime_checkable

from src.models.embedding_models.embedding_model_base import EmbeddingModelBase


@runtime_checkable
class HasEmbeddingModel(Protocol):
    """Protocol for models that contain an embedding model that can be extracted."""
    
    @property
    def embedding_model(self) -> EmbeddingModelBase | None:
        """Get the embedding model from this model."""
        ...
