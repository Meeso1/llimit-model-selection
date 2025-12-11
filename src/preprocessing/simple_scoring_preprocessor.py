"""Preprocessor for simple scoring model."""

from src.data_models.data_models import TrainingData
from src.data_models.simple_scoring_types import PreprocessedTrainingData, PreprocessedComparison
from src.utils.string_encoder import StringEncoder


class SimpleScoringPreprocessor:
    """
    Preprocessor for simple scoring model.
    
    Extracts model comparisons from training data and encodes model names to IDs.
    Handles all comparison types: model_a_wins, model_b_wins, tie, both_bad.
    """
    
    def __init__(self) -> None:
        pass
    
    def preprocess(self, data: TrainingData) -> PreprocessedTrainingData:
        """
        Preprocess training data for simple scoring model.
        
        Args:
            data: Raw training data
            
        Returns:
            Preprocessed training data with encoded model IDs
        """
        # Extract all unique model names
        model_names = set()
        for entry in data.entries:
            model_names.add(entry.model_a)
            model_names.add(entry.model_b)
        
        # Create model encoder
        model_encoder = StringEncoder()
        model_encoder.fit(sorted(model_names))
        
        # Create comparisons
        comparisons = []
        for entry in data.entries:
            model_id_a = model_encoder.encode(entry.model_a)
            model_id_b = model_encoder.encode(entry.model_b)
            
            # Map winner to comparison type
            if entry.winner == "model_a":
                comparison_type = "model_a_wins"
            elif entry.winner == "model_b":
                comparison_type = "model_b_wins"
            elif entry.winner == "tie":
                comparison_type = "tie"
            elif entry.winner == "both_bad":
                comparison_type = "both_bad"
            else:
                # Skip unknown winner types
                continue
            
            comparisons.append(PreprocessedComparison(
                model_id_a=model_id_a,
                model_id_b=model_id_b,
                comparison_type=comparison_type,
            ))
        
        return PreprocessedTrainingData(
            comparisons=comparisons,
            model_encoder=model_encoder,
        )

