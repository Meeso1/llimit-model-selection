"""Muon optimizer specification."""

from typing import Literal
import torch.nn as nn
import torch.optim as optim

from src.models.optimizers.optimizer_spec import OptimizerSpecification


# TODO: Fix
class MuonSpec(OptimizerSpecification):
    """Specification for Muon optimizer."""
    optimizer_type: Literal["muon"] = "muon"
    learning_rate: float = 0.02
    lr_decay_gamma: float | None = None
    momentum: float = 0.95
    nesterov: bool = True

    def create_optimizer_for_multiple(self, models: list[nn.Module]) -> optim.Optimizer:
        """Create Muon optimizer instance."""
        return optim.Muon(
            [param for model in models for param in model.parameters()],
            lr=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )

