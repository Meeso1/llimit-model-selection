import json
import warnings
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from typing import Any

from src.utils.jars import Jars
from src.utils.training_logger import TrainingLog, load_training_log


def run_inspect_log(args: Any) -> None:
    result = inspect_log(
        base_name=args.run_name,
        timestamp=args.timestamp,
        include_config=args.include_config,
        include_final_metrics=args.include_final_metrics,
        include_epoch_logs=args.include_epoch_logs,
    )
    print(json.dumps(result, indent=2, default=_json_default))


def inspect_log(
    base_name: str,
    timestamp: int | None = None,
    include_config: bool = True,
    include_final_metrics: bool = True,
    include_epoch_logs: bool = False,
) -> dict[str, Any]:
    log = load_training_log(base_name, timestamp=timestamp)
    return _log_to_dict(log, include_config=include_config, include_final_metrics=include_final_metrics, include_epoch_logs=include_epoch_logs)


def _log_to_dict(
    log: TrainingLog,
    include_config: bool,
    include_final_metrics: bool,
    include_epoch_logs: bool,
) -> dict[str, Any]:
    # Extract init timestamp from the full run name (base_name-{ts})
    last_dash = log.run_name.rfind("-")
    if last_dash != -1:
        run_timestamp = int(log.run_name[last_dash + 1:])
    else:
        run_timestamp = None

    base_name = log.run_name[:last_dash] if last_dash != -1 else log.run_name

    saved_model_name = getattr(log, "saved_model_name", None)
    saved_model_exists = (
        saved_model_name is not None
        and Jars.models.has_exact(saved_model_name)
    )

    result: dict[str, Any] = {
        "run_name": base_name,
        "timestamp": run_timestamp,
        "full_name": log.run_name,
        "saved_model": saved_model_exists,
    }

    if include_config:
        result["config"] = log.config

    if include_final_metrics:
        result["final_metrics"] = log.final_metrics

    if include_epoch_logs:
        result["epoch_logs"] = [
            {"epoch_index": i, "data": entry.data}
            for i, entry in enumerate(log.epoch_logs)
        ]

    return result


def _json_default(obj: Any) -> Any:
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)

    warnings.warn(f"Object of type {type(obj).__name__} is not JSON serializable, falling back to str()")
    return str(obj)
