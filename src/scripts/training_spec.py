from typing import Literal
from pydantic import BaseModel, Field, model_validator

from src.scripts.model_types import ModelSpec


DatasetType = Literal["lmarena_human_preference", "chatbot_arena", "both"]


class ModelSpecification(BaseModel):
    name: str = Field(description="Save model under this name after training")
    start_state: str | None = Field(
        default=None,
        description="If set, load latest model with given name as starting point, and ignore spec",
    )
    spec: ModelSpec | None = Field(
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
    dataset: DatasetType = Field(
        default="lmarena_human_preference",
        description="Dataset to use: 'lmarena_human_preference', 'chatbot_arena', or 'both' (combines both datasets)",
    )
    max_samples: int | None = Field(default=None, description="Sample dataset to this size before preprocessing etc. (use provided seed)")
    validation_split: float = Field(description="Data fraction to use as validation set")
    seed: int


class LoggingSpecification(BaseModel):
    """Configuration for logging and console output during training."""
    run_name: str = Field(description="Name for this training run (will be used for logging)")
    print_every: int = Field(description="Print progress every N epochs")
    # TODO: Allow to specify timing info logging here + improve how it's logged


class TrainingSpecification(BaseModel):
    model: ModelSpecification
    data: DataSpecification
    log: LoggingSpecification
    jar_base_path: str | None = Field(default=None, description="Base path for jars, if None, jars are saved in project root")
    epochs: int
    batch_size: int
