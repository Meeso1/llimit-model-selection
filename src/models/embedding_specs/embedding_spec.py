"""Base class for embedding model specifications."""

from abc import abstractmethod
from pydantic import BaseModel

from src.models.embedding_model_base import EmbeddingModelBase


class EmbeddingModelSpecification(BaseModel):
    """
    Base class for embedding model specifications.
    
    Similar to OptimizerSpecification, this allows configuring different
    embedding models (frozen vs fine-tunable vs attention-based) in a consistent way.
    """
    
    @abstractmethod
    def create_model(
        self,
        min_model_comparisons: int,
        preprocessor_seed: int,
        print_every: int | None,
    ) -> EmbeddingModelBase:
        """
        Create an embedding model instance.
        
        Args:
            min_model_comparisons: Minimum comparisons for a model to be included
            preprocessor_seed: Random seed for preprocessor
            print_every: Print progress every N epochs (None = no printing)
            
        Returns:
            Configured embedding model instance
        """
        pass
