"""Base class for optimizer specifications."""

from abc import abstractmethod
from typing import Any, Literal
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel


OptimizerType = Literal["adam", "adamw", "muon"]


class OptimizerSpecification(BaseModel):
    """
    Abstract base class for optimizer specifications.
    
    Defines how to create and serialize/deserialize optimizers.
    Each optimizer spec stores optimizer-specific parameters and can
    create optimizer instances for a given model.
    """
    optimizer_type: OptimizerType
    learning_rate: float
    lr_decay_gamma: float | None = None

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Create an optimizer instance for the given model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Configured optimizer instance
        """
        return self.create_optimizer_for_multiple([model])
    
    @abstractmethod
    def create_optimizer_for_multiple(self, models: list[nn.Module]) -> optim.Optimizer:
        """
        Create an optimizer instance for the given models.
        
        Args:
            models: List of PyTorch models to optimize
            
        Returns:
            Configured optimizer instance
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize optimizer specification to dictionary.
        
        Returns:
            Dictionary with all parameters needed to reconstruct this spec
        """
        return self.model_dump()

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
        
        match optimizer_name:
            case "adam":
                return AdamSpec.model_validate(params)
            case "adamw":
                return AdamWSpec.model_validate(params)
            case "muon":
                return MuonSpec.model_validate(params)
            case unknown:
                raise ValueError(f"Unknown optimizer name: {unknown}")
