"""Preprocessor for simple scoring model."""
from src.data_models.data_models import TrainingData
from src.data_models.simple_scoring_types import PreprocessedTrainingData, PreprocessedComparison
from src.preprocessing.utils import filter_out_rare_models, create_encoder, validate_winner_types


class SimpleScoringPreprocessor:
    """
    Preprocessor for simple scoring model.
    
    Extracts model comparisons from training data and encodes model names to IDs.
    Handles all comparison types: model_a_wins, model_b_wins, tie, both_bad.
    Optionally filters out rare models that don't have enough training data.
    """
    
    def __init__(self, min_model_occurrences: int = 1000) -> None:
        self.min_model_occurrences = min_model_occurrences
    
    def preprocess(self, data: TrainingData) -> PreprocessedTrainingData:
        """
        Preprocess training data for simple scoring model.
        
        Args:
            data: Raw training data
            
        Returns:
            Preprocessed training data with encoded model IDs
        """
        validate_winner_types(data)
        filtered_data, indexes = filter_out_rare_models(data, self.min_model_occurrences)
        model_encoder = create_encoder(filtered_data)
        
        comparisons = []
        for entry in filtered_data.entries:
            model_id_a = model_encoder.encode(entry.model_a)
            model_id_b = model_encoder.encode(entry.model_b)
            
            comparisons.append(PreprocessedComparison(
                model_id_a=model_id_a,
                model_id_b=model_id_b,
                winner=entry.winner,
            ))
        
        if len(comparisons) == 0:
            raise ValueError(
                "No training data available after filtering. "
                f"All models were filtered out (min_model_occurrences={self.min_model_occurrences}). "
                "Try lowering min_model_occurrences or providing more training data."
            )
        
        return PreprocessedTrainingData(
            comparisons=comparisons,
            model_encoder=model_encoder,
            filtered_indexes=indexes,
        )

