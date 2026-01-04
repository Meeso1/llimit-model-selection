"""Utility for tracking best model parameters during training."""

import copy
from typing import Any


class BestModelTracker:
    """Tracks and stores the best model parameters based on accuracy."""
    
    def __init__(self) -> None:
        """Initialize the tracker."""
        self.best_accuracy: float | None = None
        self._best_epoch: int | None = None
        self._best_state: dict[str, Any] | None = None
    
    def record_state(
        self,
        accuracy: float,
        module_dict: dict[str, Any],
        epoch: int
    ) -> None:
        """
        Record model state if accuracy is better than previous best.
        
        Args:
            accuracy: Current accuracy to compare
            module_dict: Dictionary of module states to save
            epoch: Current epoch number
        """
        if self.best_accuracy is None or accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self._best_epoch = epoch
            # Deep copy to avoid references to mutable objects
            self._best_state = copy.deepcopy(module_dict)
    
    def get_best_state(self) -> tuple[dict[str, Any] | None, int | None]:
        """
        Get the best recorded state.
        
        Returns:
            Tuple of (module_dict, epoch) or (None, None) if no state recorded
        """
        return self._best_state, self._best_epoch
