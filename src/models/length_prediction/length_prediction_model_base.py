"""Base class for length prediction models."""

from abc import ABC, abstractmethod
from typing import Any
import wandb

from src.utils.data_split import ValidationSplit
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.wandb_details import WandbDetails
from src.data_models.data_models import TrainingData, InputData
from src.data_models.length_prediction.length_prediction_data_models import LengthPredictionOutputData
from src.utils.jars import Jars


class LengthPredictionModelBase(ABC):
    """Base class for all length prediction models."""
    
    def __init__(self, wandb_details: WandbDetails | None = None) -> None:
        self.wandb_details = wandb_details

    def init_wandb_if_needed(self) -> None:
        if self.wandb_details is not None and self.wandb_details.init_project:
            wandb.init(
                project=self.wandb_details.project,
                name=self.wandb_details.experiment_name,
                config=self.get_config_for_wandb(),
                settings=wandb.Settings(silent=True),
            )

    def finish_wandb_if_needed(self) -> None:
        if self.wandb_details is not None and self.wandb_details.init_project:
            wandb.finish()

    @abstractmethod
    def get_config_for_wandb(self) -> dict[str, Any]:
        """Get configuration dictionary for Weights & Biases logging."""
        pass

    def log_to_wandb(self, entry: TrainingHistoryEntry) -> None:
        wandb.log(entry.to_wandb_dict())

    @abstractmethod
    def train(
        self, 
        data: TrainingData, 
        validation_split: ValidationSplit | None = None,
        epochs: int = 10, 
        batch_size: int = 32
    ) -> None:
        """
        Train the model on length prediction data.
        
        Args:
            data: Training data (same format as scoring models)
            validation_split: Optional validation split configuration
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        pass

    @abstractmethod
    def predict(
        self, 
        X: InputData, 
        batch_size: int = 32
    ) -> LengthPredictionOutputData:
        """
        Predict response lengths for given prompts and models.
        
        Args:
            X: Input data with prompts and model names (same format as scoring models)
            batch_size: Batch size for prediction
            
        Returns:
            Predicted response lengths for each prompt-model pair
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
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "LengthPredictionModelBase":
        """Load model from state dictionary."""
        pass

    def save(self, name: str) -> None:
        """Save model to disk."""
        Jars.models.add(name, self.get_state_dict())

        if self.wandb_details is not None and self.wandb_details.artifact_name is not None:
            path = Jars.models.get_latest_file_path(name)
            self._save_model_to_wandb(self.wandb_details.artifact_name, path)

    @classmethod
    def load(cls, name: str) -> "LengthPredictionModelBase":
        """Load model from disk."""
        return cls.load_state_dict(Jars.models.get(name))

    def _save_model_to_wandb(self, name: str, path: str) -> None:
        artifact = wandb.Artifact(name=name, type="model", description="Length prediction model state dict")
        artifact.add_file(path)

        logged_artifact = wandb.log_artifact(artifact)
        logged_artifact.wait()
