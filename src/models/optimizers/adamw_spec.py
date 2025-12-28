"""AdamW optimizer specification."""

from typing import Literal
import torch.nn as nn
import torch.optim as optim

from src.models.optimizers.optimizer_spec import OptimizerSpecification


class AdamWSpec(OptimizerSpecification):
    """Specification for AdamW optimizer."""
    optimizer_type: Literal["adamw"] = "adamw"
    learning_rate: float = 0.001
    lr_decay_gamma: float | None = None
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    def create_optimizer_for_multiple(self, models: list[nn.Module]) -> optim.Optimizer:
        """Create AdamW optimizer instance."""
        return optim.AdamW(
            [param for model in models for param in model.parameters()],
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

