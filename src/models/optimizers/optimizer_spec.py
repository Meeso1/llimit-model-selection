"""Base class for optimizer specifications."""

from abc import ABC, abstractmethod
from typing import Any
import torch.nn as nn
import torch.optim as optim


class OptimizerSpecification(ABC):
    """
    Abstract base class for optimizer specifications.
    
    Defines how to create and serialize/deserialize optimizers.
    Each optimizer spec stores optimizer-specific parameters and can
    create optimizer instances for a given model.
    """

    def __init__(
        self,
        learning_rate: float,
        lr_decay_gamma: float | None = None,
    ) -> None:
        """
        Initialize optimizer specification.
        
        Args:
            learning_rate: Base learning rate
            lr_decay_gamma: Exponential LR decay factor (multiplied each epoch).
                           If None, no decay is applied.
        """
        self.learning_rate = learning_rate
        self.lr_decay_gamma = lr_decay_gamma

    @abstractmethod
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Create an optimizer instance for the given model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Configured optimizer instance
        """
        pass

    @abstractmethod
    def get_optimizer_name(self) -> str:
        """
        Get the optimizer name for serialization.
        
        Returns:
            String identifier for this optimizer type
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize optimizer specification to dictionary.
        
        Returns:
            Dictionary with all parameters needed to reconstruct this spec
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, params: dict[str, Any]) -> "OptimizerSpecification":
        """
        Deserialize optimizer specification from dictionary.
        
        Args:
            params: Dictionary with parameters from to_dict()
            
        Returns:
            Reconstructed optimizer specification
        """
        pass

    def create_scheduler(
        self,
        optimizer: optim.Optimizer,
    ) -> optim.lr_scheduler.LRScheduler | None:
        """
        Create learning rate scheduler if decay is enabled.
        
        Args:
            optimizer: Optimizer instance to schedule
            
        Returns:
            Scheduler instance or None if no decay
        """
        if self.lr_decay_gamma is None:
            return None
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay_gamma)

    @staticmethod
    def from_serialized(optimizer_name: str, params: dict[str, Any]) -> "OptimizerSpecification":
        """
        Create optimizer specification from serialized data.
        
        Args:
            optimizer_name: Name of optimizer type
            params: Serialized parameters dictionary
            
        Returns:
            OptimizerSpecification instance
            
        Raises:
            ValueError: If optimizer name is not recognized
        """
        # Import here to avoid circular dependencies
        from src.models.optimizers.adam_spec import AdamSpec
        from src.models.optimizers.adamw_spec import AdamWSpec
        from src.models.optimizers.muon_spec import MuonSpec
        
        optimizer_classes = {
            "adam": AdamSpec,
            "adamw": AdamWSpec,
            "muon": MuonSpec,
        }
        
        if optimizer_name not in optimizer_classes:
            raise ValueError(
                f"Unknown optimizer name: {optimizer_name}. "
                f"Available: {list(optimizer_classes.keys())}"
            )
        
        return optimizer_classes[optimizer_name].from_dict(params)
