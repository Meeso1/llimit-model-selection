from pydantic import BaseModel, Field, model_validator

from src.scripts.model_types import ModelSpecBase
from src.utils.wandb_details import WandbDetails


class WandbDetailsSpecification(BaseModel):
    project: str
    experiment_name: str
    config_name: str
    artifact_name: str

    def to_wandb_details(self) -> WandbDetails:
        return WandbDetails(
            project=self.project,
            experiment_name=self.experiment_name,
            config_name=self.config_name,
            artifact_name=self.artifact_name,
            init_project=True,
        )


class ModelSpecification(BaseModel):
    name: str = Field(description="Save model under this name after training")
    start_state: str | None = Field(
        default=None,
        description="If set, load latest model with given name as starting point, and ignore spec",
    )
    spec: ModelSpecBase | None = Field(
        default=None,
        description="Model-specific specification, used when creating new model (translated to constructor arguments)",
    )

    @model_validator(mode="after")
    def validate_start_state_or_spec(self) -> "ModelSpecification":
        """Validate that either start_state or spec is provided, but not both."""
        if self.start_state is None and self.spec is None:
            raise ValueError("Either 'start_state' or 'spec' must be provided in model specification")
        if self.start_state is not None and self.spec is not None:
            raise ValueError("Cannot specify both 'start_state' and 'spec' in model specification")
        return self


class DataSpecification(BaseModel):
    max_samples: int = Field(description="Sample dataset to this size before preprocessing etc. (use provided seed)")
    valiation_split: float = Field(description="Data fraction to use as validation set")
    seed: int


class LoggingSpecification(BaseModel):
    print_every: int
    # TODO: Allow to specify timing info logging here + improve how it's logged
    # TODO: Extend - allow to log to .jsonl file?
    # Probably best to have some `Logger` class and use that instead of the method in model


class TrainingSpecification(BaseModel):
    wandb: WandbDetailsSpecification
    model: ModelSpecification
    data: DataSpecification
    log: LoggingSpecification
    epochs: int
    batch_size: int
