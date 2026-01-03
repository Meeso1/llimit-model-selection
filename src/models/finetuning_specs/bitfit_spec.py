"""BitFit fine-tuning specification."""

from typing import Literal
import torch.nn as nn

from src.models.finetuning_specs.finetuning_spec import FineTuningSpecification


class BitFitSpec(FineTuningSpecification):
    """
    BitFit fine-tuning specification.
    
    Only trains bias terms, freezing all other parameters.
    Extremely parameter-efficient (~0.1% of model parameters).
    Surprisingly effective for some tasks.
    """
    
    method: Literal["bitfit"] = "bitfit"
    
    def apply_to_model(self, model: nn.Module, quiet: bool = False) -> nn.Module:
        """Freeze all parameters except biases."""
        trainable_count = 0
        total_count = 0
        
        for name, param in model.named_parameters():
            total_count += param.numel()
            if "bias" in name:
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
        
        if not quiet:
            print(f"BitFit fine-tuning applied (bias terms only):")
            print(f"  Trainable params: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.2f}%)")
        
        return model

