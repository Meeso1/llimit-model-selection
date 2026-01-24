"""Inference service for the API."""

from src.data_models.data_models import InputData
from src.models.model_base import ModelBase
from src.scripts.model_types import ModelType
from src.scripts import infer


class InferenceService:
    """Service for running model inference."""
    
    def __init__(self) -> None:
        """Initialize the inference service."""
        self._loaded_model_name: str | None = None
        self._loaded_model_type: ModelType | None = None
        self._loaded_model: ModelBase | None = None
    
    def infer(
        self,
        model_type: ModelType,
        model_name: str,
        models_to_score: list[str],
        prompts: list[str],
        batch_size: int,
    ) -> dict[str, list[float]]:
        """
        Run inference on a trained model.
        
        Args:
            model_type: Type of model to load
            model_name: Name of the saved model
            models_to_score: List of model names to score
            prompts: List of prompts to evaluate
            batch_size: Batch size for inference
            
        Returns:
            Dictionary mapping model names to their scores for each prompt
        """
        model = self._get_or_load_model(model_type, model_name)
        
        input_data = InputData(prompts=prompts, model_names=models_to_score)
        result = model.predict(input_data, batch_size)
        
        scores_dict = {
            model_name: scores.tolist()
            for model_name, scores in result.scores.items()
        }
        
        return scores_dict
    
    def _get_or_load_model(self, model_type: ModelType, model_name: str) -> ModelBase:
        if self._loaded_model_type == model_type \
            and self._loaded_model_name == model_name \
            and self._loaded_model is not None:
            return self._loaded_model
        
        self._loaded_model = infer.load_model(model_type, model_name)
        self._loaded_model_type = model_type
        self._loaded_model_name = model_name
        return self._loaded_model
