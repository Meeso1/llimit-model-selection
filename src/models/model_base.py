from abc import ABC, abstractmethod
from typing import Any, ClassVar
import wandb

from src.constants import MODELS_JAR_PATH
from src.utils.jar import Jar
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.wandb_details import WandbDetails
from src.data_models.data_models import InputData, OutputData, TrainingData


class ModelBase(ABC):
    models_jar: ClassVar[Jar] = Jar(MODELS_JAR_PATH)
    
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
        pass

    def log_to_wandb(self, entry: TrainingHistoryEntry) -> None:
        wandb.log(entry.to_wandb_dict())

    @abstractmethod
    def train(self, data: TrainingData, epochs: int = 10, batch_size: int = 32) -> None:
        # Preprocess data or use cached preprocessed data if available
        # Train model
        pass

    @abstractmethod
    def predict(self, X: InputData, batch_size: int = 32) -> OutputData:
        pass

    @abstractmethod
    def get_history(self) -> TrainingHistory:
        pass

    @abstractmethod
    def get_state_dict(self) -> dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "ModelBase":
        pass

    def save(self, name: str) -> None:
        self.models_jar.add(name, self.get_state_dict())

        if self.wandb_details is not None and self.wandb_details.artifact_name is not None:
            path = self.models_jar.get_latest_file_path(name)
            self._save_model_to_wandb(self.wandb_details.artifact_name, path)

    @classmethod
    def load(cls, name: str) -> "ModelBase":
        return cls.load_state_dict(cls.models_jar.get(name))

    def _save_model_to_wandb(self, name: str, path: str) -> None:
        artifact = wandb.Artifact(name=name, type="model", description="Model state dict")
        artifact.add_file(path)

        logged_artifact = wandb.log_artifact(artifact)
        logged_artifact.wait()
