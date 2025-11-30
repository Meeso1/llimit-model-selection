from dataclasses import dataclass
import numpy as np
from src.data_models.data_models import TrainingData
from src.data_models.dense_network_types import PreprocessedTrainingData


@dataclass
class ValidationSplit:
    """Configuration for train/validation split."""
    val_fraction: float = 0.2
    seed: int = 42


def train_val_split(
    data: TrainingData,
    val_fraction: float = 0.2,
    seed: int = 42
) -> tuple[TrainingData, TrainingData]:
    """
    Splits training data into train and validation sets deterministically.
    
    Args:
        data: The training data to split
        val_fraction: Fraction of data to use for validation (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (train_data, val_data) as TrainingData objects
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    
    n_total = len(data.entries)
    if n_total == 0:
        raise ValueError("Cannot split empty training data")
    
    rng = np.random.RandomState(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    
    n_val = int(n_total * val_fraction)
    if n_val == 0:
        raise ValueError(f"Validation set would be empty. Dataset has {n_total} entries and val_fraction={val_fraction}")
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_entries = [data.entries[i] for i in train_indices]
    val_entries = [data.entries[i] for i in val_indices]
    
    return TrainingData(entries=train_entries), TrainingData(entries=val_entries)


def split_preprocessed_data(
    preprocessed_data: PreprocessedTrainingData,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[PreprocessedTrainingData, PreprocessedTrainingData]:
    """
    Split preprocessed training data into train and validation sets.
    
    This operates on already-preprocessed pairs, maintaining the same model encoder.
    
    Args:
        preprocessed_data: Preprocessed training data to split
        val_fraction: Fraction of data to use for validation (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (train_preprocessed, val_preprocessed) with shared model encoder
    """
    if val_fraction == 0:
        return preprocessed_data, None
    
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    
    n_total = len(preprocessed_data.pairs)
    if n_total == 0:
        raise ValueError("Cannot split empty preprocessed data")
    
    rng = np.random.RandomState(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    
    n_val = int(n_total * val_fraction)
    if n_val == 0:
        raise ValueError(f"Validation set would be empty. Dataset has {n_total} pairs and val_fraction={val_fraction}")
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_pairs = [preprocessed_data.pairs[i] for i in train_indices]
    val_pairs = [preprocessed_data.pairs[i] for i in val_indices]
    
    train_preprocessed = PreprocessedTrainingData(
        pairs=train_pairs,
        embedding_dim=preprocessed_data.embedding_dim,
        model_encoder=preprocessed_data.model_encoder,
    )
    
    val_preprocessed = PreprocessedTrainingData(
        pairs=val_pairs,
        embedding_dim=preprocessed_data.embedding_dim,
        model_encoder=preprocessed_data.model_encoder,
    )
    
    return train_preprocessed, val_preprocessed

