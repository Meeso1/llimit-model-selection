"""Muon optimizer specification."""

from typing import Literal
import torch.nn as nn
import torch.optim as optim

from src.models.optimizers.optimizer_spec import OptimizerSpecification
from src.models.optimizers.muon_optimizer import HybridMuonOptimizer


class MuonSpec(OptimizerSpecification):
    """
    Specification for Muon optimizer with AdamW fallback.
    
    Creates a hybrid optimizer that uses:
    - torch.optim.Muon for ≥2D transformer parameters (weight matrices)
    - torch.optim.AdamW for everything else (biases, layer norms, scoring heads)
    
    Manually separates parameters to apply different learning rates and optimizers
    to different components, which is beneficial for transfer learning scenarios.
    
    Supports optimizing parameters from multiple models simultaneously.
    """
    optimizer_type: Literal["muon"] = "muon"
    learning_rate: float = 0.02
    lr_decay_gamma: float | None = None
    momentum: float = 0.95
    nesterov: bool = True
    
    # AdamW parameters for non-Muon params
    adamw_lr: float = 0.0003
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)

    def create_optimizer_for_multiple(
        self,
        models: list[nn.Module],
        lr_multipliers: dict[nn.Module, float] | None = None,
    ) -> optim.Optimizer:
        """
        Create hybrid Muon+AdamW optimizer instance for one or more models.
        
        Args:
            models: List of models to optimize
            lr_multipliers: Optional dict mapping specific modules to LR multipliers
            
        Returns:
            HybridMuonOptimizer instance
        """
        return HybridMuonOptimizer(
            models=models,
            muon_lr=self.learning_rate,
            adamw_lr=self.adamw_lr,
            muon_momentum=self.momentum,
            muon_nesterov=self.nesterov,
            adamw_weight_decay=self.weight_decay,
            adamw_betas=self.betas,
            lr_multipliers=lr_multipliers,
        )

