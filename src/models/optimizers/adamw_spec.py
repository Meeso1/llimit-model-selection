"""AdamW optimizer specification."""

from typing import Any
import torch.nn as nn
import torch.optim as optim

from src.models.optimizers.optimizer_spec import OptimizerSpecification


class AdamWSpec(OptimizerSpecification):
    """Specification for AdamW optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr_decay_gamma: float | None = None,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """
        Initialize AdamW optimizer specification.
        
        Args:
            learning_rate: Learning rate
            lr_decay_gamma: Exponential LR decay factor (if None, no decay)
            betas: Coefficients for computing running averages
            eps: Term added to denominator for numerical stability
            weight_decay: Decoupled weight decay coefficient
        """
        super().__init__(learning_rate, lr_decay_gamma)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create AdamW optimizer instance."""
        return optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

    def get_optimizer_name(self) -> str:
        """Get optimizer name."""
        return "adamw"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "lr_decay_gamma": self.lr_decay_gamma,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        }

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "AdamWSpec":
        """Deserialize from dictionary."""
        return cls(
            learning_rate=params["learning_rate"],
            lr_decay_gamma=params.get("lr_decay_gamma"),
            betas=tuple(params["betas"]),
            eps=params["eps"],
            weight_decay=params["weight_decay"],
        )

