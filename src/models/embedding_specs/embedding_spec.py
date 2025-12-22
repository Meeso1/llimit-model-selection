"""Base class for embedding model specifications."""

from abc import abstractmethod
from pydantic import BaseModel

from src.models.triplet_model_base import TripletModelBase


class EmbeddingModelSpecification(BaseModel):
    """
    Base class for embedding model specifications.
    
    Similar to OptimizerSpecification, this allows configuring different
    embedding models (frozen vs fine-tunable) in a consistent way.
    """
    
    triplet_margin: float = 0.2
    regularization_weight: float = 0.01
    identity_positive_ratio: float = 0.8
    
    @abstractmethod
    def create_model(
        self,
        min_model_comparisons: int,
        preprocessor_seed: int,
        print_every: int | None,
    ) -> TripletModelBase:
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
