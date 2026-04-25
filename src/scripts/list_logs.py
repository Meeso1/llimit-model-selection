import json
from typing import Any

from src.utils.training_logger import list_training_runs


def run_list_logs(args: Any) -> None:
    list_timestamps: bool = args.list_timestamps
    as_json: bool = args.json
    list_logs(list_timestamps=list_timestamps, as_json=as_json)


def list_logs(list_timestamps: bool = False, as_json: bool = False) -> None:
    runs = list_training_runs()

    if as_json:
        output = [
            {
                "base_name": run.base_name,
                "timestamps": sorted([int(ts.timestamp()) for ts in run.timestamps], reverse=True),
                "latest_timestamp": int(run.latest_timestamp.timestamp()),
                "latest_full_name": run.latest_full_name,
            }
            for run in runs
        ]
        print(json.dumps(output, indent=2))
        return

    max_base_name_length = max(len(run.base_name) for run in runs) if runs else 0
    for run in runs:
        print(f"{run.latest_timestamp}: {run.base_name.ljust(max_base_name_length)} - {len(run.timestamps)} versions, latest at {run.latest_timestamp}")
        if list_timestamps:
            for timestamp in run.timestamps:
                print(f"\t{int(timestamp.timestamp())}")