"""Preprocessor for simple scoring model."""

from collections import Counter
from src.data_models.data_models import TrainingData
from src.data_models.simple_scoring_types import PreprocessedTrainingData, PreprocessedComparison
from src.utils.string_encoder import StringEncoder


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
        model_counts = Counter()
        for entry in data.entries:
            model_counts[entry.model_a] += 1
            model_counts[entry.model_b] += 1
        
        frequent_models = {
            model_name 
            for model_name, count in model_counts.items() 
            if count >= self.min_model_occurrences
        }
        
        model_encoder = StringEncoder()
        model_encoder.fit(sorted(frequent_models))
        
        comparisons = []
        for entry in data.entries:
            if entry.model_a not in frequent_models or entry.model_b not in frequent_models:
                continue
            
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
        
        if len(comparisons) == 0:
            raise ValueError(
                "No training data available after filtering. "
                f"All models were filtered out (min_model_occurrences={self.min_model_occurrences}). "
                "Try lowering min_model_occurrences or providing more training data."
            )
        
        return PreprocessedTrainingData(
            comparisons=comparisons,
            model_encoder=model_encoder,
        )

