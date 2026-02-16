"""Utility for tracking best model parameters during training."""

import copy
from typing import Any


class BestModelTracker:
    """Tracks and stores the best model parameters based on accuracy."""
    
    def __init__(self) -> None:
        """Initialize the tracker."""
        self.best_accuracy: float | None = None
        self.best_epoch: int | None = None
        self._best_state_dict: dict[str, Any] | None = None
    
    def record_state(
        self,
        accuracy: float,
        state_dict: dict[str, Any],
        epoch: int
    ) -> None:
        """
        Record model state dict if accuracy is better than previous best.
        
        Args:
            accuracy: Current accuracy to compare
            state_dict: Full model state dictionary to save
            epoch: Current epoch number
        """
        if self.best_accuracy is None or accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = epoch
            # Deep copy to avoid references to mutable objects
            self._best_state_dict = copy.deepcopy(state_dict)
    
    @property
    def best_state_dict(self) -> dict[str, Any] | None:
        """
        Get the best recorded state dict.
        
        Returns:
            Best state dictionary or None if no state recorded
        """
        return self._best_state_dict
    
    @property
    def has_best_state(self) -> bool:
        """Check if a best state has been recorded."""
        return self._best_state_dict is not None
