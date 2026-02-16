"""Common base class for all models."""

from abc import ABC, abstractmethod
from typing import Any, Literal, Self

from src.utils.data_split import ValidationSplit
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.training_logger import TrainingLogger
from src.data_models.data_models import TrainingData, InputData
from src.utils.jars import Jars


ModelKind = Literal["scoring", "length_prediction"]

class ModelBase[TOutput](ABC):
    """
    Common base class for all models (scoring and length prediction).
    
    Provides common functionality for training, prediction, serialization, and logging.
    """
    
    def __init__(self, run_name: str | None = None) -> None:
        """
        Initialize the model.
        
        Args:
            run_name: Name for logging this training run. If None, logging is disabled.
        """
        if run_name is not None:
            self._logger: TrainingLogger | None = TrainingLogger(run_name)
        else:
            self._logger = None

    @property
    @abstractmethod
    def model_kind(self) -> ModelKind:
        """Return the kind of model (scoring or length_prediction)."""
        pass

    def init_logger_if_needed(self) -> None:
        """Initialize training logger if configured."""
        if self._logger is not None:
            self._logger.init(config=self.get_config_for_logging())

    def finish_logger_if_needed(self, final_metrics: dict[str, Any] | None = None) -> None:
        """
        Finish training logger and optionally log final metrics.
        
        Args:
            final_metrics: Optional dictionary of final metrics to log
        """
        if self._logger is not None:
            if final_metrics is not None:
                self._logger.log_final_metrics(final_metrics)
            self._logger.finish()

    @abstractmethod
    def get_config_for_logging(self) -> dict[str, Any]:
        """Get configuration dictionary for training logging."""
        pass

    def append_entry_to_log(self, entry: TrainingHistoryEntry) -> None:
        """Log a training history entry."""
        if self._logger is not None:
            self._logger.log(entry.to_dict())

    @abstractmethod
    def train(
        self, 
        data: TrainingData, 
        validation_split: ValidationSplit | None = None,
        epochs: int = 10, 
        batch_size: int = 32
    ) -> None:
        """Train the model on data."""
        pass

    @abstractmethod
    def predict(self, X: InputData, batch_size: int = 32) -> TOutput:
        """
        Predict on input data.
        
        Returns OutputData (for scoring models) or LengthPredictionOutputData (for length prediction models).
        """
        pass

    @abstractmethod
    def get_history(self) -> TrainingHistory:
        """Get training history."""
        pass

    @abstractmethod
    def get_state_dict(self) -> dict[str, Any]:
        """Get model state dictionary for saving."""
        pass

    @classmethod
    @abstractmethod
    def load_state_dict(cls, state_dict: dict[str, Any], instance: Self | None = None) -> Self:
        """
        Load model from state dictionary.
        
        Args:
            state_dict: State dictionary to load
            instance: Optional existing model instance to load into. If provided,
                     loads state into this instance instead of creating a new one.
                     Must be of the same type as cls.
        
        Returns:
            The loaded model instance (either newly created or the provided instance)
        """
        pass

    def save(self, name: str) -> None:
        """Save model to disk."""
        Jars.models.add(name, self.get_state_dict())

    @classmethod
    def load(cls, name: str) -> Self:
        """Load model from disk."""
        return cls.load_state_dict(Jars.models.get(name))
