from typing import Any

from src.utils.training_logger import list_training_runs


def run_list_logs(args: Any) -> None:
    list_timestamps: bool = args.list_timestamps
    list_logs(list_timestamps=list_timestamps)


def list_logs(list_timestamps: bool = False) -> None:
    runs = list_training_runs()

    max_base_name_length = max(len(run.base_name) for run in runs) if runs else 0
    for run in runs:
        print(f"{run.latest_timestamp}: {run.base_name.ljust(max_base_name_length)} - {len(run.timestamps)} versions, latest at {run.latest_timestamp}")
        if list_timestamps:
            for timestamp in run.timestamps:
                print(f"\t{int(timestamp.timestamp())}")