"""Base class for fine-tuning specifications."""

from abc import abstractmethod
from typing import Any, Literal
from pydantic import BaseModel
import torch.nn as nn

FineTuningMethod = Literal["full", "last_layers", "lora", "qlora", "bitfit"]


class FineTuningSpecification(BaseModel):
    """
    Base class for fine-tuning method specifications.
    
    Similar to OptimizerSpecification and EmbeddingModelSpecification,
    this allows configuring different fine-tuning methods in a consistent way.
    """
    
    method: FineTuningMethod
    
    def get_quantization_config(self):
        """
        Get quantization config for loading the base model.
        
        Returns:
            BitsAndBytesConfig if quantization is needed, None otherwise
        """
        return None
    
    @abstractmethod
    def apply_to_model(self, model: nn.Module, quiet: bool = False) -> nn.Module:
        """
        Apply fine-tuning configuration to a model.
        
        Args:
            model: Base transformer model
            quiet: Whether to suppress print statements

        Returns:
            Model with fine-tuning method applied (may be wrapped with PEFT adapters)
        """
        pass
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize specification to dictionary."""
        return self.model_dump()
    
    @staticmethod
    def from_serialized(method: str, params: dict[str, Any]) -> "FineTuningSpecification":
        """
        Create fine-tuning specification from serialized data.
        
        Args:
            method: Name of fine-tuning method
            params: Serialized parameters dictionary
            
        Returns:
            FineTuningSpecification instance
            
        Raises:
            ValueError: If method name is not recognized
        """
        # Import here to avoid circular dependencies
        from src.models.finetuning_specs.lora_spec import LoraSpec
        from src.models.finetuning_specs.qlora_spec import QLoraSpec
        from src.models.finetuning_specs.last_layers_spec import LastLayersSpec
        from src.models.finetuning_specs.bitfit_spec import BitFitSpec
        from src.models.finetuning_specs.full_finetuning_spec import FullFineTuningSpec
        
        match method:
            case "lora":
                return LoraSpec.model_validate(params)
            case "qlora":
                return QLoraSpec.model_validate(params)
            case "last_layers":
                return LastLayersSpec.model_validate(params)
            case "bitfit":
                return BitFitSpec.model_validate(params)
            case "full":
                return FullFineTuningSpec.model_validate(params)
            case _:
                raise ValueError(f"Unknown fine-tuning method: {method}")

