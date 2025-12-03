from dataclasses import dataclass
from typing import Any, Literal

from src.scripts.model_types import ModelSpecBase, ModelType
from src.utils.wandb_details import WandbDetails


@dataclass
class WandbDetailsSpecification:
    project: str
    experiment_name: str
    config_name: str
    artifact_name: str

    def to_wandb_details(self):
        return WandbDetails(
            project=self.project,
            experiment_name=self.experiment_name,
            config_name=self.config_name,
            artifact_name=self.artifact_name,
            init_project=True,
        )


@dataclass
class ModelSpecification:
    name: str # Save model under this name after training
    start_state: str | None # If set, load latest model with given name as starting point, and ignore spec
    # TODO: Ensure that spec type is correctly validated 
    spec: ModelSpecBase | None # Model-specific specificatiion, used when creating new model (translated to constructor arguments)

@dataclass
class DataSpecification:
    max_samples: int # Sample dataset to this size before preprocessing etc. (use provided seed)
    valiation_split: float # Data fraction to use as valiadtion set
    seed : int


@dataclass
class LoggingSpecification:
    print_every: int
    # TODO: Allow to specify timing info logging here + improve how it's logged
    # TODO: Extend - allow to log to .jsonl file? 
    # Probably best to have some `Logger` class and use that instead of the method in model


# TODO: Parse this from JSON
@dataclass
class TrainingSpecification:
    wandb: WandbDetailsSpecification
    model: ModelSpecification
    data: DataSpecification
    log: LoggingSpecification
    epochs: int
    batch_size: int
