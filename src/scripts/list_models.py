from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.utils.jars import Jars


def run_list_models(args: Any) -> None:
    list_checkpoints: bool = args.list_checkpoints
    list_models(list_checkpoints=list_checkpoints)


# TODO: this prints model checkpoints separately, since they have '@[epoch]' suffix - improve
def list_models(list_checkpoints: bool = False) -> None:
    all_model_names = Jars.models.object_names()
    timestamps_per_model = {
        model_name: Jars.models.get_all_version_timestamps(model_name)
        for model_name in all_model_names
    }

    non_checkpoint_entries, checkpoints_per_model = _extract_checkpoint_entries(timestamps_per_model)

    all_model_names = set(non_checkpoint_entries.keys()) | set(checkpoints_per_model.keys())
    details_per_model = [
        _get_model_details(model_name, non_checkpoint_entries.get(model_name, []), checkpoints_per_model.get(model_name, {}))
        for model_name in all_model_names
    ]

    details_per_model.sort(key=lambda x: x.latest_timestamp, reverse=True)

    max_model_name_length = max(len(details.model_name) for details in details_per_model) if details_per_model else 0

    for details in details_per_model:
        print(f"{details.latest_timestamp}: {details.model_name.ljust(max_model_name_length)} - {details.num_versions} versions, {details.num_checkpoints} checkpoints, {details.num_distinct_checkpoints} distinct")
        if list_checkpoints:
            for checkpoint_name, checkpoint_timestamps in details.checkpoints.items():
                print(f"\t{checkpoint_name}: {len(checkpoint_timestamps)} versions, latest at {max(checkpoint_timestamps)}")


def _extract_checkpoint_entries(
    timestamps_per_model: dict[str, list[datetime]]
) -> tuple[dict[str, list[datetime]], dict[str, dict[str, list[datetime]]]]:
    checkpoint_separator = "@ep"

    checkpoint_entries = {}
    other_entries = {}
    for model_name, timestamps in timestamps_per_model.items():
        if checkpoint_separator in model_name and model_name.rsplit(checkpoint_separator, 1)[1].isdigit():
            checkpoint_entries[model_name] = timestamps
        else:
            other_entries[model_name] = timestamps

    checkpoints_per_model = {}
    for entry_name, values in checkpoint_entries.items():
        model_name_without_checkpoint = entry_name.rsplit(checkpoint_separator, 1)[0]
        if model_name_without_checkpoint not in checkpoints_per_model:
            checkpoints_per_model[model_name_without_checkpoint] = {}

        checkpoints_per_model[model_name_without_checkpoint][entry_name] = values

    return other_entries, checkpoints_per_model


def _get_model_details(model_name: str, versions: list[datetime], checkpoints: dict[str, list[datetime]]) -> "ModelDetails":
    num_versions = len(versions)
    num_distinct_checkpoints = len(checkpoints)
    num_checkpoints = sum(len(timestamps) for timestamps in checkpoints.values())

    all_timestamps = versions
    for checkpoint_timestamps in checkpoints.values():
        all_timestamps.extend(checkpoint_timestamps)

    latest_timestamp = max(all_timestamps)

    return ModelDetails(
        model_name=model_name,
        num_versions=num_versions,
        num_checkpoints=num_checkpoints,
        num_distinct_checkpoints=num_distinct_checkpoints,
        latest_timestamp=latest_timestamp,
        checkpoints=checkpoints
    )


@dataclass
class ModelDetails:
    model_name: str
    num_versions: int
    num_checkpoints: int
    num_distinct_checkpoints: int
    latest_timestamp: datetime
    checkpoints: dict[str, list[datetime]]
