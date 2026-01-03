"""Last-layers fine-tuning specification."""

from typing import Literal
import torch.nn as nn

from src.models.finetuning_specs.finetuning_spec import FineTuningSpecification


class LastLayersSpec(FineTuningSpecification):
    """
    Last-layers fine-tuning specification.
    
    Freezes all layers except the last N transformer encoder layers.
    Simple and effective, no external dependencies required.
    """
    
    method: Literal["last_layers"] = "last_layers"
    num_unfrozen_layers: int = 2
    
    def apply_to_model(self, model: nn.Module, quiet: bool = False) -> nn.Module:
        """Freeze all layers except last N encoder layers."""
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze last N encoder layers
        encoder_layers = self._get_encoder_layers(model)
        for layer in encoder_layers[-self.num_unfrozen_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Always unfreeze pooler if present
        if hasattr(model, 'pooler') and model.pooler is not None:
            for param in model.pooler.parameters():
                param.requires_grad = True
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        if not quiet:
            print(f"Last-layers fine-tuning applied (last {self.num_unfrozen_layers} layers unfrozen):")
            print(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
        
        return model
    
    def _get_encoder_layers(self, model: nn.Module) -> nn.ModuleList:
        """Get encoder layers handling different model architectures."""
        # BERT, RoBERTa, etc.
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            return model.encoder.layer
        # DistilBERT
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
            return model.transformer.layer
        # GPT-2
        if hasattr(model, 'h'):
            return model.h
        # Fallback: try common patterns
        for attr in ['layers', 'blocks', 'encoder_layers']:
            if hasattr(model, attr):
                return getattr(model, attr)
        raise ValueError(f"Could not find encoder layers in model {type(model)}")

