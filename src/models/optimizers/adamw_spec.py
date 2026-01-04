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

    def create_optimizer_for_multiple(
        self,
        models: list[nn.Module],
        lr_multipliers: dict[nn.Module, float] | None = None,
    ) -> optim.Optimizer:
        """
        Create AdamW optimizer instance with optional per-module learning rate multipliers.
        
        Args:
            models: List of models to optimize
            lr_multipliers: Optional dict mapping specific modules to LR multipliers
            
        Returns:
            AdamW optimizer instance
        """
        all_params = [param for model in models for param in model.parameters()]
        
        if lr_multipliers is None or len(lr_multipliers) == 0 or all(multiplier == 1.0 for multiplier in lr_multipliers.values()):
            return optim.AdamW(
                all_params,
                lr=self.learning_rate,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        
        # Build parameter -> multiplier lookup
        param_to_multiplier: dict[nn.Parameter, float] = {}
        for module, multiplier in lr_multipliers.items():
            for param in module.parameters():
                if param in param_to_multiplier:
                    raise ValueError(
                        f"Parameter appears in multiple modules with different LR multipliers. "
                        f"This is likely due to shared parameters or overlapping module specifications."
                    )
                param_to_multiplier[param] = multiplier
        
        # Group parameters by their effective learning rate
        lr_to_params: dict[float, list[nn.Parameter]] = {}
        for model in models:
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                
                multiplier = param_to_multiplier.get(param, 1.0)
                effective_lr = self.learning_rate * multiplier
                
                if effective_lr not in lr_to_params:
                    lr_to_params[effective_lr] = []
                lr_to_params[effective_lr].append(param)
        
        param_groups = [
            {'params': params, 'lr': lr}
            for lr, params in lr_to_params.items()
        ]
        
        return optim.AdamW(
            param_groups,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

