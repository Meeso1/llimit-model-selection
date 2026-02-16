import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data_models.data_models import InputData
from src.models.length_prediction.length_prediction_model_base import LengthPredictionModelBase
from src.models.model_loading import ModelType, load_model
from src.constants import INFERENCE_OUTPUTS_PATH
from src.models.scoring.scoring_model_base import ScoringModelBase


def run_infer(args: Any) -> None:
    """
    Run inference from command line arguments.
    
    Args:
        args: Parsed command line arguments with model_type, model_name, model_names,
              prompts, batch_size, and optional output_path
    """
    if "/" not in args.model:
        raise ValueError("Model must be of the form 'model_type/model_name'")

    model_type, model_name = args.model.split("/")
    models_to_score: list[str] = args.models_to_score
    requested_prompts: list[str] = args.prompts
    batch_size: int = args.batch_size
    
    output_path = args.output_path or _make_default_output_path(model_name)
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
    models_to_score: list[str],
    requested_prompts: list[str], 
    batch_size: int,
    output_path: str,
) -> None:
    model = load_model(model_type, model_name)
    input_data = InputData(prompts=requested_prompts, model_names=models_to_score)

    # Handle different output types based on model kind
    if model.model_kind == "scoring":
        output_data = _infer_scoring(ScoringModelBase.assert_kind(model), input_data, batch_size)
    elif model.model_kind == "length_prediction":
        output_data = _infer_length_prediction(LengthPredictionModelBase.assert_kind(model), input_data, batch_size)
    else:
        raise ValueError(f"Unknown model kind: {model.model_kind}")
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Inference results ({output_data['result_type']}) saved to: {output_path}")


def _infer_scoring(model: ScoringModelBase, input_data: InputData, batch_size: int) -> dict[str, list[float]]:
    result = model.predict(input_data, batch_size)
    result_dict = {model_name: scores.tolist() for model_name, scores in result.scores.items()}
    return {
        "result_type": "scores",
        "results": result_dict,
    }


def _infer_length_prediction(model: LengthPredictionModelBase, input_data: InputData, batch_size: int) -> dict[str, list[float]]:
    result = model.predict(input_data, batch_size)
    result_dict = {model_name: lengths.tolist() for model_name, lengths in result.predictions.items()}
    return {
        "result_type": "predicted_lengths",
        "results": result_dict,
    }


def _make_default_output_path(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    output_filename = f"{timestamp}_{safe_model_name}.json"
    return str(INFERENCE_OUTPUTS_PATH / output_filename)
