from typing import Any

from src.data_models.data_models import InputData
from src.models.dense_network_model import DenseNetworkModel
from src.models.model_base import ModelBase
from src.scripts.model_types import ModelType


def run_infer(args: Any) -> None:
    # TODO: implement
    # Get relevant stuff from args
    # Validate
    # Call `infer()`
    pass


def infer(
    model_type: ModelType, 
    model_name: str,
    requested_model_names: list[str],
    requested_prompts: list[str], 
    batch_size: int
) -> None:
    model = _load_model(model_type, model_name)

    input_data = InputData(prompts=requested_prompts, model_names=requested_model_names)
    result =  model.predict(input_data, batch_size)

    # TODO: Save output somewhere


def _load_model(model_type: ModelType, model_name: str) -> ModelBase:
    match model_type:
        case "dense_network":
            return DenseNetworkModel.load(model_name)
        case unknown:
            raise ValueError(f"Unknown model type: {unknown}")  # pyright: ignore[reportUnreachable]
