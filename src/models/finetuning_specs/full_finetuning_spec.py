"""Full fine-tuning specification."""

from typing import Literal
import torch.nn as nn

from src.models.finetuning_specs.finetuning_spec import FineTuningSpecification


class FullFineTuningSpec(FineTuningSpecification):
    """
    Full fine-tuning specification.
    
    Trains all parameters in the model (no freezing).
    Most parameter-intensive but potentially highest quality.
    Requires most memory and compute.
    """
    
    method: Literal["full"] = "full"
    
    def apply_to_model(self, model: nn.Module, quiet: bool = False) -> nn.Module:
        """Unfreeze all parameters."""
        for param in model.parameters():
            param.requires_grad = True
        
        total = sum(p.numel() for p in model.parameters())
        
        if not quiet:
            print(f"Full fine-tuning applied (all parameters trainable):")
            print(f"  Trainable params: {total:,} / {total:,} (100.00%)")
        
        return model

