"""Adam optimizer specification."""

from typing import Literal
import torch.nn as nn
import torch.optim as optim

from src.models.optimizers.optimizer_spec import OptimizerSpecification


class AdamSpec(OptimizerSpecification):
    """Specification for Adam optimizer."""
    optimizer_type: Literal["adam"] = "adam"
    learning_rate: float = 0.001
    lr_decay_gamma: float | None = None
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create Adam optimizer instance."""
        return optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

