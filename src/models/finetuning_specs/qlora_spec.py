"""QLoRA (Quantized LoRA) fine-tuning specification."""

from typing import Literal
import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.models.finetuning_specs.finetuning_spec import FineTuningSpecification


class QLoraSpec(FineTuningSpecification):
    """
    QLoRA (Quantized LoRA) fine-tuning specification.
    
    LoRA with 4-bit or 8-bit quantization of the base model.
    Allows fine-tuning much larger models on limited VRAM.
    
    Note: Model must be loaded with quantization config BEFORE applying QLoRA.
    This spec only configures LoRA on top of the quantized model.
    """
    
    method: Literal["qlora"] = "qlora"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bits: Literal[4, 8] = 4
    target_modules: list[str] | Literal["auto"] = "auto"
    
    def get_quantization_config(self):
        """Get quantization config for loading the base model."""
        if self.bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Invalid bits value: {self.bits}")
    
    def apply_to_model(self, model: nn.Module, quiet: bool = False) -> nn.Module:
        """Apply LoRA to the quantized model."""
        model = prepare_model_for_kbit_training(model)
        
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
            print(f"QLoRA applied successfully ({self.bits}-bit, rank={self.rank}, alpha={self.alpha}):")
            model.print_trainable_parameters()
        
        return model

