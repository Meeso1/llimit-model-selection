"""Hybrid optimizer that combines Muon (for ≥2D params) with AdamW (for other params)."""

from typing import Any
import torch.nn as nn
import torch.optim as optim


class HybridMuonOptimizer:
    """
    Hybrid optimizer that uses Muon for ≥2D transformer parameters
    and AdamW for everything else (biases, norms, scoring head).
    
    Uses torch.optim.Muon for transformer weight matrices and torch.optim.AdamW
    for other parameters. This allows different learning rates for different
    components (e.g., pre-trained vs newly initialized).
    
    Supports optimizing parameters from multiple models simultaneously.
    """
    
    def __init__(
        self,
        models: list[nn.Module],
        muon_lr: float = 0.02,
        adamw_lr: float = 0.0003,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        adamw_weight_decay: float = 0.01,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        lr_multipliers: dict[nn.Module, float] | None = None,
    ) -> None:
        """
        Initialize hybrid optimizer.
        
        Args:
            models: List of models to optimize
            muon_lr: Learning rate for Muon (transformer weight matrices)
            adamw_lr: Learning rate for AdamW (everything else)
            muon_momentum: Momentum for Muon
            muon_nesterov: Whether to use Nesterov momentum in Muon
            adamw_weight_decay: Weight decay for AdamW
            adamw_betas: Beta parameters for AdamW
            lr_multipliers: Optional dict mapping specific modules to LR multipliers (applied to AdamW params)
        """        
        # Build parameter -> multiplier lookup
        param_to_multiplier: dict[nn.Parameter, float] = {}
        if lr_multipliers:
            for module, multiplier in lr_multipliers.items():
                for param in module.parameters():
                    if param in param_to_multiplier:
                        raise ValueError(
                            f"Parameter appears in multiple modules with different LR multipliers. "
                            f"This is likely due to shared parameters or overlapping module specifications."
                        )
                    param_to_multiplier[param] = multiplier
        
        # Separate parameters into Muon-eligible and AdamW-eligible
        muon_params = []  # ≥2D parameters in transformer
        adamw_lr_to_params: dict[float, list[nn.Parameter]] = {}  # Group by effective LR
        
        for model in models:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Use Muon for ≥2D transformer parameters
                if 'transformer' in name and param.ndim >= 2:
                    muon_params.append(param)
                else:
                    # AdamW parameters: apply multiplier if specified
                    multiplier = param_to_multiplier.get(param, 1.0)
                    effective_lr = adamw_lr * multiplier
                    
                    if effective_lr not in adamw_lr_to_params:
                        adamw_lr_to_params[effective_lr] = []
                    adamw_lr_to_params[effective_lr].append(param)
        
        # Create Muon optimizer using torch.optim.Muon
        if len(muon_params) > 0:
            self.muon: optim.Optimizer | None = optim.Muon(
                muon_params,
                lr=muon_lr,
                momentum=muon_momentum,
                nesterov=muon_nesterov,
            )
            print(f"Muon optimizer: {len(muon_params)} parameters (≥2D transformer weights) @ lr={muon_lr}")
        else:
            self.muon = None
        
        # Create AdamW with parameter groups for different learning rates
        if len(adamw_lr_to_params) > 0:
            adamw_param_groups = [
                {'params': params, 'lr': lr}
                for lr, params in sorted(adamw_lr_to_params.items())
            ]
            
            self.adamw: optim.Optimizer = optim.AdamW(
                adamw_param_groups,
                betas=adamw_betas,
                weight_decay=adamw_weight_decay,
            )
            
            # Print info about parameter groups
            total_adamw_params = sum(len(params) for params in adamw_lr_to_params.values())
            print(f"AdamW optimizer: {total_adamw_params} parameters (biases, norms, heads)")
            for lr, params in sorted(adamw_lr_to_params.items()):
                if lr != adamw_lr:
                    multiplier = lr / adamw_lr
                    print(f"  - {len(params)} parameters @ lr={lr:.6f} (multiplier={multiplier:.1f})")
        else:
            # Shouldn't happen, but create empty optimizer as fallback
            self.adamw = optim.AdamW([], lr=adamw_lr)
    
    def step(self) -> None:
        """Perform optimization step for both optimizers."""
        if self.muon is not None:
            self.muon.step()
        self.adamw.step()
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients for both optimizers."""
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> dict[str, Any]:
        """Return state dictionary containing both optimizer states."""
        state = {
            'adamw': self.adamw.state_dict(),
        }
        if self.muon is not None:
            state['muon'] = self.muon.state_dict()
        return state
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary for both optimizers."""
        if 'adamw' in state_dict:
            self.adamw.load_state_dict(state_dict['adamw'])
        if self.muon is not None and 'muon' in state_dict:
            self.muon.load_state_dict(state_dict['muon'])

