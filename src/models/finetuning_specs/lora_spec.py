"""LoRA (Low-Rank Adaptation) fine-tuning specification."""

from typing import Literal
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from src.models.finetuning_specs.finetuning_spec import FineTuningSpecification


class LoraSpec(FineTuningSpecification):
    """
    LoRA (Low-Rank Adaptation) fine-tuning specification.
    
    Adds small trainable low-rank decomposition matrices alongside frozen weights.
    Very parameter-efficient (0.1-1% of model parameters trainable).
    """
    
    method: Literal["lora"] = "lora"
    rank: int = 16
    alpha: int = 32  # Scaling factor, typically 2x rank
    dropout: float = 0.05
    target_modules: list[str] | Literal["auto"] = "auto"
    
    def apply_to_model(self, model: nn.Module, quiet: bool = False) -> nn.Module:
        """Apply LoRA to the model."""
        # Auto-detect target modules if not specified
        target = None if self.target_modules == "auto" else self.target_modules
        
        config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=target,
            bias="none",
        )
        model = get_peft_model(model, config)
        
        if not quiet:
            print(f"LoRA applied successfully (rank={self.rank}, alpha={self.alpha}):")
            model.print_trainable_parameters()
        
        return model

