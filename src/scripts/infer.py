import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data_models.data_models import InputData
from src.models.dense_network_model import DenseNetworkModel
from src.models.model_base import ModelBase
from src.scripts.model_types import ModelType
from src.constants import INFERENCE_OUTPUTS_PATH


def run_infer(args: Any) -> None:
    """
    Run inference from command line arguments.
    
    Args:
        args: Parsed command line arguments with model_type, model_name, model_names,
              prompts, batch_size, and optional output_path
    """
    model_type: ModelType = args.model_type
    model_name: str = args.model_name
    models_to_score: list[str] = args.models_to_score
    requested_prompts: list[str] = args.prompts
    batch_size: int = args.batch_size
    
    # Construct output path
    if args.output_path:
        output_path = args.output_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
        output_filename = f"{timestamp}_{safe_model_name}.json"
        output_path = str(INFERENCE_OUTPUTS_PATH / output_filename)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    infer(
        model_type=model_type,
        model_name=model_name,
        models_to_score=models_to_score,
        requested_prompts=requested_prompts,
        batch_size=batch_size,
        output_path=output_path,
    )


def infer(
    model_type: ModelType, 
    model_name: str,
    requested_model_names: list[str],
    requested_prompts: list[str], 
    batch_size: int,
    output_path: str,
) -> None:
    model = _load_model(model_type, model_name)

    input_data = InputData(prompts=requested_prompts, model_names=requested_model_names)
    result =  model.predict(input_data, batch_size)

    result_dict = {model_name: scores.tolist() for model_name, scores in result.scores.items()}

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=4)


def _load_model(model_type: ModelType, model_name: str) -> ModelBase:
    match model_type:
        case "dense_network":
            return DenseNetworkModel.load(model_name)
        case unknown:
            raise ValueError(f"Unknown model type: {unknown}")  # pyright: ignore[reportUnreachable]
