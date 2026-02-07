"""Inference service for the API."""

from src.data_models.data_models import InputData
from src.models.length_prediction.length_prediction_model_base import LengthPredictionModelBase
from src.models.scoring_model_base import ScoringModelBase
from src.models.model_loading import LengthPredictionModelType, load_length_prediction_model, ScoringModelType, load_scoring_model


class InferenceService:
    """Service for running model inference."""
    
    def __init__(self) -> None:
        """Initialize the inference service."""
        self._loaded_scoring_model_name: str | None = None
        self._loaded_scoring_model_type: ScoringModelType | None = None
        self._loaded_scoring_model: ScoringModelBase | None = None
        
        self._loaded_length_prediction_model_name: str | None = None
        self._loaded_length_prediction_model_type: LengthPredictionModelType | None = None
        self._loaded_length_prediction_model: LengthPredictionModelBase | None = None
    
    def score(
        self,
        model_type: ScoringModelType,
        model_name: str,
        models_to_score: list[str],
        prompts: list[str],
        batch_size: int,
    ) -> dict[str, list[float]]:
        """
        Run inference on a trained scoring model.
        
        Args:
            model_type: Type of scoring model to load
            model_name: Name of the saved model
            models_to_score: List of model names to score
            prompts: List of prompts to evaluate
            batch_size: Batch size for inference
            
        Returns:
            Dictionary mapping model names to their scores for each prompt
        """
        model = self._get_or_load_scoring_model(model_type, model_name)
        
        input_data = InputData(prompts=prompts, model_names=models_to_score)
        result = model.predict(input_data, batch_size)
        
        scores_dict = {
            model_name: scores.tolist()
            for model_name, scores in result.scores.items()
        }
        
        return scores_dict
    
    def predict_lengths(
        self,
        model_type: LengthPredictionModelType,
        model_name: str,
        model_names: list[str],
        prompts: list[str],
        batch_size: int,
    ) -> dict[str, list[float]]:
        """
        Run length prediction on a trained length prediction model.
        
        Args:
            model_type: Type of model to load
            model_name: Name of the saved model
            model_names: List of model names to predict lengths for
            prompts: List of prompts to evaluate
            batch_size: Batch size for inference
            
        Returns:
            Dictionary mapping model names to their predicted lengths for each prompt
        """
        model = self._get_or_load_length_prediction_model(model_type, model_name)
        
        input_data = InputData(prompts=prompts, model_names=model_names)
        result = model.predict(input_data, batch_size)
        
        lengths_dict = {
            model_name: lengths.tolist()
            for model_name, lengths in result.predicted_lengths.items()
        }
        
        return lengths_dict
    
    def _get_or_load_scoring_model(self, model_type: ScoringModelType, model_name: str) -> ScoringModelBase:
        if self._loaded_scoring_model_type == model_type \
            and self._loaded_scoring_model_name == model_name \
            and self._loaded_scoring_model is not None:
            return self._loaded_scoring_model
        
        self._loaded_scoring_model = load_scoring_model(model_type, model_name)
        self._loaded_scoring_model_type = model_type
        self._loaded_scoring_model_name = model_name
        return self._loaded_scoring_model

    def _get_or_load_length_prediction_model(self, model_type: LengthPredictionModelType, model_name: str) -> LengthPredictionModelBase:
        if self._loaded_length_prediction_model_type == model_type \
            and self._loaded_length_prediction_model_name == model_name \
            and self._loaded_length_prediction_model is not None:
            return self._loaded_length_prediction_model
        
        self._loaded_length_prediction_model = load_length_prediction_model(model_type, model_name)
        self._loaded_length_prediction_model_type = model_type
        self._loaded_length_prediction_model_name = model_name
        return self._loaded_length_prediction_model
