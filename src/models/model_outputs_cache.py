"""Caching wrapper for model predictions."""

import warnings

from src.models.scoring.scoring_model_base import ScoringModelBase
from src.data_models.data_models import EvaluationEntry, InputData, OutputData
from src.utils.timer import Timer


class ModelOutputsCache:
    """
    Caches predictions from a model.
    """
    
    def __init__(
        self,
        model: ScoringModelBase,
        quiet: bool = False,
    ) -> None:
        """
        Initialize the model cache.
        
        Args:
            model: The model to cache predictions from
            quiet: Whether to suppress output
        """
        self.model = model
        self.quiet = quiet
        self._scores: dict[int, tuple[float, float]] | None = None
    
    def compute_and_cache(
        self,
        entries: list[EvaluationEntry],
        indexes: list[int],
        timer: Timer | None = None,
    ) -> None:
        """
        Compute base model predictions for training data and cache them.
        
        Args:
            data: Training data
            indexes: Indexes of the data entries to cache
            timer: Optional parent timer for tracking
        """
        assert len(entries) == len(indexes), "Number of data entries and indexes must match"

        with Timer("compute_base_model_cache", verbosity="start+end", parent=timer):
            self._scores = {}

            if not self.quiet:
                print(f"Computing base model predictions for {len(entries)} pairs...")
            
            for entry, index in zip(entries, indexes):
                input_data = InputData(prompts=[entry.user_prompt], model_names=[entry.model_a, entry.model_b])
                result = self.model.predict(input_data)

                if index in self._scores:
                    warnings.warn(f"Duplicate index {index} was provided")

                self._scores[index] = (result.scores[entry.model_a][0], result.scores[entry.model_b][0])
            
            if not self.quiet:
                print(f"Cached predictions for {len(self._scores)} pairs")
    
    def get_base_scores(
        self,
        indexes: list[int],
    ) -> tuple[list[float], list[float]]:
        """
        Get cached base model scores for given data indices.
        
        Args:
            indexes: Indices in the data
            
        Returns:
            Tuple of (scores_a, scores_b) lists
        """
        if self._scores is None:
            raise RuntimeError("Cache not computed. Call compute_and_cache first.")
        
        scores_a = []
        scores_b = []

        for idx in indexes:
            score_a, score_b = self._scores[idx]
            scores_a.append(score_a)
            scores_b.append(score_b)
        
        return scores_a, scores_b

    def predict(
        self, 
        input_data: InputData,
        batch_size: int = 32,
    ) -> OutputData:
        return self.model.predict(input_data, batch_size=batch_size)
