from typing import Any
import datasets
import numpy as np

from src import data_loading
from src.data_models.data_models import TrainingData
from src.models.dense_network_model import DenseNetworkModel
from src.models.model_base import ModelBase
from src.scripts.model_types import DenseNetworkSpecification
from src.scripts.training_spec import TrainingSpecification


def run_train(args: Any) -> None:
    # TODO: implement
    # Parse input (stdin? file with provided path?) to TrainingSpecification
    # Validate
    # Call `train()`
    pass


def train(spec: TrainingSpecification) -> None:
    model = _create_starting_model(spec)

    training_data = _load_lmarena_human_preference()
    downsampled = _downsample(training_data, spec.data.max_samples, spec.data.seed)

    model.train(downsampled, spec.epochs, spec.batch_size)
    model.save(spec.model.name)


def _create_starting_model(spec: TrainingSpecification) -> ModelBase:
    match spec.model.spec.model_type:
        case "dense_network":
            return _create_starting_dense_network(spec)
        case unknown:
            raise ValueError(f"Unknown model type: {unknown}")  # pyright: ignore[reportUnreachable]


def _create_starting_dense_network(training_spec: TrainingSpecification) -> DenseNetworkModel:
    if not isinstance(training_spec.model.spec, DenseNetworkSpecification):
        raise ValueError(f"Expected model specification to be of type {DenseNetworkSpecification.__name__}, but found {type(training_spec.model.spec).__name__}")
    
    model_spec = training_spec.model.spec
    if training_spec.model.name is not None:
        return DenseNetworkModel.load(training_spec.model.name)

    return DenseNetworkModel(
        embedding_model_name=model_spec.embedding_model_name,
        hidden_dims=model_spec.hidden_dims,
        model_id_embedding_dim=model_spec.model_id_embedding_dim,
        learning_rate=model_spec.optimizer.learning_rate,
        wandb_details=training_spec.wandb.to_wandb_details(),
    )


def _load_lmarena_human_preference() -> TrainingData:
    dataset = datasets.load_dataset("lmarena-ai/arena-human-preference-140k")
    training_data = data_loading.load_training_data(dataset["train"].to_pandas())
    print(f"Successfully loaded {len(training_data.entries)} entries")
    return training_data


# TODO: move to 'utils/data_split.py'
def _downsample(data: TrainingData, max_samples: int, seed: int) -> TrainingData:
    # TODO: implement
    pass
