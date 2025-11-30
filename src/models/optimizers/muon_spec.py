"""Muon optimizer specification."""

from typing import Any
import torch.nn as nn
import torch.optim as optim

from src.models.optimizers.optimizer_spec import OptimizerSpecification


# TODO: Fix
class MuonSpec(OptimizerSpecification):
    """Specification for Muon optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.02,
        lr_decay_gamma: float | None = None,
        momentum: float = 0.95,
        nesterov: bool = True,
    ) -> None:
        """
        Initialize Muon optimizer specification.
        
        Args:
            learning_rate: Learning rate
            lr_decay_gamma: Exponential LR decay factor (if None, no decay)
            momentum: Momentum factor
            nesterov: Whether to use Nesterov momentum
        """
        super().__init__(learning_rate, lr_decay_gamma)
        self.momentum = momentum
        self.nesterov = nesterov

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create Muon optimizer instance."""
        return optim.Muon(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )

    def get_optimizer_name(self) -> str:
        """Get optimizer name."""
        return "muon"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "lr_decay_gamma": self.lr_decay_gamma,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
        }

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "MuonSpec":
        """Deserialize from dictionary."""
        return cls(
            learning_rate=params["learning_rate"],
            lr_decay_gamma=params.get("lr_decay_gamma"),
            momentum=params["momentum"],
            nesterov=params["nesterov"],
        )

