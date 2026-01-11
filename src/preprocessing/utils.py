from collections import Counter
import warnings
from src.data_models.data_models import TrainingData
from src.utils.string_encoder import StringEncoder


def validate_winner_types(data: TrainingData) -> None:
    for entry in data.entries:
        if entry.winner not in ["model_a", "model_b", "tie", "both_bad"]:
            warnings.warn(f"Unknown winner type: {entry.winner}")


def filter_out_ties(data: TrainingData, indexes: list[int] | None = None) -> tuple[TrainingData, list[int]]:
    if indexes is None:
        indexes = list(range(len(data.entries)))
        
    assert len(indexes) == len(data.entries)
    
    return TrainingData(entries=[entry for entry in data.entries if entry.winner != "tie"]), \
        [original for i, original in enumerate(indexes) if data.entries[i].winner != "tie"]


def filter_out_both_bad(data: TrainingData, indexes: list[int] | None = None) -> tuple[TrainingData, list[int]]:
    if indexes is None:
        indexes = list(range(len(data.entries)))
        
    assert len(indexes) == len(data.entries)
    
    return TrainingData(entries=[entry for entry in data.entries if entry.winner != "both_bad"]), \
        [original for i, original in enumerate(indexes) if data.entries[i].winner != "both_bad"]


def filter_out_rare_models(data: TrainingData, min_model_occurrences: int = 1000, indexes: list[int] | None = None) -> tuple[TrainingData, list[int]]:
    """
    Filter out models with insufficient comparisons.
    
    Args:
        data: Training data
        min_model_occurrences: Minimum number of comparisons for a model to be included
        indexes: Indexes of the entries to filter

    Returns:
        Tuple of (filtered_data, filtered_indexes)
    """
    if indexes is None:
        indexes = list(range(len(data.entries)))
        
    assert len(indexes) == len(data.entries)
    
    model_counts = Counter()
    for entry in data.entries:
        model_counts[entry.model_a] += 1
        model_counts[entry.model_b] += 1

    frequent_models = {
        model_name 
        for model_name, count in model_counts.items() 
        if count >= min_model_occurrences
    }
    
    filtered_entries = []
    filtered_indexes = []
    for i, entry in zip(indexes, data.entries):
        if entry.model_a in frequent_models and entry.model_b in frequent_models:
            filtered_entries.append(entry)
            filtered_indexes.append(i)

    return TrainingData(entries=filtered_entries), filtered_indexes


def filter_out_empty_entries(data: TrainingData, indexes: list[int] | None = None) -> tuple[TrainingData, list[int]]:
    if indexes is None:
        indexes = list(range(len(data.entries)))
        
    assert len(indexes) == len(data.entries)
    
    filtered_entries = []
    filtered_indexes = []
    for i, entry in zip(indexes, data.entries):
        if entry.user_prompt.strip() and entry.model_a_response.strip() and entry.model_b_response.strip():
            filtered_entries.append(entry)
            filtered_indexes.append(i)
    
    return TrainingData(entries=filtered_entries), filtered_indexes


def create_encoder(data: TrainingData) -> StringEncoder:
    model_names = [name for entry in data.entries for name in [entry.model_a, entry.model_b]]
    model_encoder = StringEncoder()
    model_encoder.fit(model_names)
    return model_encoder