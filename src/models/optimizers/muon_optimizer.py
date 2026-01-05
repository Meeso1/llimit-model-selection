"""Hybrid optimizer that combines Muon (for ≥2D params) with AdamW (for other params)."""

from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim


class HybridMuonOptimizer(optim.Optimizer):
    """
    Hybrid optimizer that uses Muon for ≥2D transformer parameters
    and AdamW for everything else (biases, norms, scoring head).
    
    Uses torch.optim.Muon for transformer weight matrices and torch.optim.AdamW
    for other parameters. This allows different learning rates for different
    components (e.g., pre-trained vs newly initialized).
    
    Supports optimizing parameters from multiple models simultaneously.
    
    Inherits from torch.optim.Optimizer to properly work with LR schedulers
    and other PyTorch optimization tooling.
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
            self._muon: optim.Optimizer | None = optim.Muon(
                muon_params,
                lr=muon_lr,
                momentum=muon_momentum,
                nesterov=muon_nesterov,
            )
            print(f"Muon optimizer: {len(muon_params)} parameters (≥2D transformer weights) @ lr={muon_lr}")
        else:
            self._muon = None
        
        # Create AdamW with parameter groups for different learning rates
        if len(adamw_lr_to_params) > 0:
            adamw_param_groups = [
                {'params': params, 'lr': lr}
                for lr, params in sorted(adamw_lr_to_params.items())
            ]
            
            self._adamw: optim.Optimizer = optim.AdamW(
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
            self._adamw = optim.AdamW([], lr=adamw_lr)
        
        # Collect all param groups for the parent Optimizer class
        # We need to initialize the parent with all parameters
        param_groups = []
        if self._muon is not None:
            param_groups.extend(self._muon.param_groups)
        param_groups.extend(self._adamw.param_groups)
        
        # Set defaults (used by parent class)
        defaults = {
            'lr': adamw_lr,  # Default LR (most params use AdamW)
        }
        
        # Initialize parent Optimizer class
        # We pass the param groups directly since we've already created them
        super().__init__(param_groups, defaults)
    
    def _sync_lr_to_internal_optimizers(self) -> None:
        """
        Sync learning rates from self.param_groups (managed by parent/schedulers)
        to internal optimizers' param_groups.
        """
        # Sync LR changes from parent param_groups to internal optimizers
        muon_groups_count = len(self._muon.param_groups) if self._muon is not None else 0
        
        for i, group in enumerate(self.param_groups):
            if i < muon_groups_count:
                # This is a Muon param group
                self._muon.param_groups[i]['lr'] = group['lr']
            else:
                # This is an AdamW param group
                adamw_idx = i - muon_groups_count
                self._adamw.param_groups[adamw_idx]['lr'] = group['lr']
    
    def step(self, closure=None) -> None:
        """Perform optimization step for both optimizers."""
        # Sync learning rates from parent param_groups to internal optimizers
        # (LR schedulers modify parent's param_groups)
        self._sync_lr_to_internal_optimizers()
        
        # Perform optimization step on both internal optimizers
        if self._muon is not None:
            self._muon.step()
        self._adamw.step()
        
        # Handle closure if provided (required by Optimizer interface)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return loss
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for both optimizers."""
        if self._muon is not None:
            self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self) -> dict[str, Any]:
        """Return state dictionary containing both optimizer states."""
        # Get parent state dict (includes param_groups and state)
        parent_state = super().state_dict()
        
        # Add internal optimizer states for full recovery
        parent_state['_internal_optimizers'] = {
            'adamw': self._adamw.state_dict(),
        }
        if self._muon is not None:
            parent_state['_internal_optimizers']['muon'] = self._muon.state_dict()
        
        return parent_state
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary for both optimizers."""
        # Load parent state (param_groups and state)
        super().load_state_dict(state_dict)
        
        # Load internal optimizer states if available
        if '_internal_optimizers' in state_dict:
            internal = state_dict['_internal_optimizers']
            if 'adamw' in internal:
                self._adamw.load_state_dict(internal['adamw'])
            if self._muon is not None and 'muon' in internal:
                self._muon.load_state_dict(internal['muon'])
        
        # Sync LRs to internal optimizers
        self._sync_lr_to_internal_optimizers()

