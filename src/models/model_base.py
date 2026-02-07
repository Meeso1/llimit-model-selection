"""Common base class for all models."""

from abc import ABC, abstractmethod
from typing import Any, Literal
import wandb

from src.utils.data_split import ValidationSplit
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.wandb_details import WandbDetails
from src.data_models.data_models import TrainingData, InputData
from src.utils.jars import Jars


ModelKind = Literal["scoring", "length_prediction"]

class ModelBase[TOutput](ABC):
    """
    Common base class for all models (scoring and length prediction).
    
    Provides common functionality for training, prediction, serialization, and WandB integration.
    """
    
    def __init__(self, wandb_details: WandbDetails | None = None) -> None:
        self.wandb_details = wandb_details

    @property
    @abstractmethod
    def model_kind(self) -> ModelKind:
        """Return the kind of model (scoring or length_prediction)."""
        pass

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
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "ModelBase":
        """Load model from state dictionary."""
        pass

    def save(self, name: str) -> None:
        """Save model to disk."""
        Jars.models.add(name, self.get_state_dict())

        if self.wandb_details is not None and self.wandb_details.artifact_name is not None:
            path = Jars.models.get_latest_file_path(name)
            self._save_model_to_wandb(self.wandb_details.artifact_name, path)

    @classmethod
    def load(cls, name: str) -> "ModelBase":
        """Load model from disk."""
        return cls.load_state_dict(Jars.models.get(name))

    def _save_model_to_wandb(self, name: str, path: str) -> None:
        artifact = wandb.Artifact(name=name, type="model", description="Model state dict")
        artifact.add_file(path)

        logged_artifact = wandb.log_artifact(artifact)
        logged_artifact.wait()
