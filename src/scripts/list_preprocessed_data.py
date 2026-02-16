from typing import Any
from src.utils.jars import Jars


def run_list_preprocessed_data(args: Any) -> None:
    # Left for consistency with commands that need arguments
    list_preprocessed_data()


def list_preprocessed_data() -> None:
    all_preprocessed_data = Jars.preprocessed_data.object_names()
    timestamps_per_preprocessed_data = {
        name: Jars.preprocessed_data.get_all_version_timestamps(name)
        for name in all_preprocessed_data
    }

    details_per_preprocessed_data = [
        (name, len(timestamps), max(timestamps))
        for name, timestamps in timestamps_per_preprocessed_data.items()
    ]
    details_per_preprocessed_data.sort(key=lambda x: x[2], reverse=True)

    max_preprocessed_data_name_length = max(len(details[0]) for details in details_per_preprocessed_data) if details_per_preprocessed_data else 0

    for name, num_versions, latest_timestamp in details_per_preprocessed_data:
        print(f"{latest_timestamp}: {name.ljust(max_preprocessed_data_name_length)} - {num_versions} versions, latest at {latest_timestamp}")