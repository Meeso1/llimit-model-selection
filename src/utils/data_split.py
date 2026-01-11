from dataclasses import dataclass
import numpy as np
from typing import TypeVar
from src.data_models.triplet_encoder_types import PreprocessedTripletEncoderData
from src.data_models.data_models import TrainingData
from src.data_models.dense_network_types import PreprocessedTrainingData as DenseNetworkPreprocessedTrainingData
from src.data_models.dn_embedding_network_types import PreprocessedTrainingData as DnEmbeddingPreprocessedTrainingData
from src.data_models.transformer_embedding_types import PreprocessedTrainingData as TransformerEmbeddingPreprocessedTrainingData
from src.data_models.simple_scoring_types import PreprocessedTrainingData as SimplePreprocessedTrainingData
from src.data_models.attention_embedding_types import ModelSetSample, PreprocessedAttentionEmbeddingData

T = TypeVar('T')


@dataclass
class ValidationSplit:
    """Configuration for train/validation split."""
    val_fraction: float = 0.2
    seed: int = 42


def _compute_split_indices(
    n_total: int,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:  # [n_train], [n_val]
    """
    Compute train and validation indices for a dataset split.
    
    This function ensures that given the same n_total, val_fraction, and seed,
    the exact same indices will be selected for train and validation sets.
    
    Args:
        n_total: Total number of items in the dataset
        val_fraction: Fraction of data to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_indices, val_indices) as numpy arrays
    
    Raises:
        ValueError: If val_fraction is not in (0, 1) or if validation set would be empty
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    
    if n_total == 0:
        raise ValueError("Cannot split empty dataset (n_total=0)")
    
    n_val = int(n_total * val_fraction)
    if n_val == 0:
        raise ValueError(f"Validation set would be empty. Dataset has {n_total} entries and val_fraction={val_fraction}")
    
    rng = np.random.RandomState(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    return train_indices, val_indices


def train_val_split(
    data: TrainingData,
    val_fraction: float = 0.2,
    seed: int = 42
) -> tuple[TrainingData, TrainingData | None]:
    """
    Splits training data into train and validation sets deterministically.
    
    Args:
        data: The training data to split
        val_fraction: Fraction of data to use for validation (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (train_data, val_data) as TrainingData objects
    """
    if val_fraction == 0:
        return data, None
    
    n_total = len(data.entries)
    train_indices, val_indices = _compute_split_indices(n_total, val_fraction, seed)
    
    train_entries = [data.entries[i] for i in train_indices]
    val_entries = [data.entries[i] for i in val_indices]
    
    return TrainingData(entries=train_entries), TrainingData(entries=val_entries)


def split_dense_network_preprocessed_data(
    preprocessed_data: DenseNetworkPreprocessedTrainingData,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[DenseNetworkPreprocessedTrainingData, DenseNetworkPreprocessedTrainingData]:
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
    
    n_total = len(preprocessed_data.pairs)
    train_indices, val_indices = _compute_split_indices(n_total, val_fraction, seed)
    
    train_pairs = [preprocessed_data.pairs[i] for i in train_indices]
    val_pairs = [preprocessed_data.pairs[i] for i in val_indices]
    
    train_indexes = [preprocessed_data.filtered_indexes[i] for i in train_indices]
    val_indexes = [preprocessed_data.filtered_indexes[i] for i in val_indices]
    
    train_preprocessed = DenseNetworkPreprocessedTrainingData(
        pairs=train_pairs,
        embedding_dim=preprocessed_data.embedding_dim,
        prompt_features_dim=preprocessed_data.prompt_features_dim,
        model_encoder=preprocessed_data.model_encoder,
        filtered_indexes=train_indexes,
    )
    
    val_preprocessed = DenseNetworkPreprocessedTrainingData(
        pairs=val_pairs,
        embedding_dim=preprocessed_data.embedding_dim,
        prompt_features_dim=preprocessed_data.prompt_features_dim,
        model_encoder=preprocessed_data.model_encoder,
        filtered_indexes=val_indexes,
    )
    
    return train_preprocessed, val_preprocessed


def split_dn_embedding_preprocessed_data(
    preprocessed_data: DnEmbeddingPreprocessedTrainingData,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[DnEmbeddingPreprocessedTrainingData, DnEmbeddingPreprocessedTrainingData]:
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
    
    n_total = len(preprocessed_data.pairs)
    train_indices, val_indices = _compute_split_indices(n_total, val_fraction, seed)
    
    train_pairs = [preprocessed_data.pairs[i] for i in train_indices]
    val_pairs = [preprocessed_data.pairs[i] for i in val_indices]
    
    train_preprocessed = DnEmbeddingPreprocessedTrainingData(
        pairs=train_pairs,
        prompt_features_dim=preprocessed_data.prompt_features_dim,
    )
    
    val_preprocessed = DnEmbeddingPreprocessedTrainingData(
        pairs=val_pairs,
        prompt_features_dim=preprocessed_data.prompt_features_dim,
    )
    
    return train_preprocessed, val_preprocessed


def split_transformer_embedding_preprocessed_data(
    preprocessed_data: TransformerEmbeddingPreprocessedTrainingData,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[TransformerEmbeddingPreprocessedTrainingData, TransformerEmbeddingPreprocessedTrainingData | None]:
    """
    Split preprocessed transformer embedding training data into train and validation sets.
    
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
    
    n_total = len(preprocessed_data.pairs)
    train_indices, val_indices = _compute_split_indices(n_total, val_fraction, seed)
    
    train_pairs = [preprocessed_data.pairs[i] for i in train_indices]
    val_pairs = [preprocessed_data.pairs[i] for i in val_indices]
    
    train_indexes = [preprocessed_data.filtered_indexes[i] for i in train_indices]
    val_indexes = [preprocessed_data.filtered_indexes[i] for i in val_indices]
    
    train_preprocessed = TransformerEmbeddingPreprocessedTrainingData(
        pairs=train_pairs,
        prompt_features_dim=preprocessed_data.prompt_features_dim,
        model_encoder=preprocessed_data.model_encoder,
        filtered_indexes=train_indexes,
    )
    
    val_preprocessed = TransformerEmbeddingPreprocessedTrainingData(
        pairs=val_pairs,
        prompt_features_dim=preprocessed_data.prompt_features_dim,
        model_encoder=preprocessed_data.model_encoder,
        filtered_indexes=val_indexes,
    )
    
    return train_preprocessed, val_preprocessed


def split_simple_scoring_preprocessed_data(
    preprocessed_data: SimplePreprocessedTrainingData,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[SimplePreprocessedTrainingData, SimplePreprocessedTrainingData | None]:
    """
    Split preprocessed simple scoring data into train and validation sets.
    
    This operates on already-preprocessed comparisons, maintaining the same model encoder.
    
    Args:
        preprocessed_data: Preprocessed training data to split
        val_fraction: Fraction of data to use for validation (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (train_preprocessed, val_preprocessed) with shared model encoder.
        val_preprocessed is None if val_fraction is 0.
    """
    if val_fraction == 0:
        return preprocessed_data, None
    
    n_total = len(preprocessed_data.comparisons)
    train_indices, val_indices = _compute_split_indices(n_total, val_fraction, seed)
    
    train_comparisons = [preprocessed_data.comparisons[i] for i in train_indices]
    val_comparisons = [preprocessed_data.comparisons[i] for i in val_indices]
    
    train_indexes = [preprocessed_data.filtered_indexes[i] for i in train_indices]
    val_indexes = [preprocessed_data.filtered_indexes[i] for i in val_indices]
    
    train_preprocessed = SimplePreprocessedTrainingData(
        comparisons=train_comparisons,
        model_encoder=preprocessed_data.model_encoder,
        filtered_indexes=train_indexes,
    )
    
    val_preprocessed = SimplePreprocessedTrainingData(
        comparisons=val_comparisons,
        model_encoder=preprocessed_data.model_encoder,
        filtered_indexes=val_indexes,
    )
    
    return train_preprocessed, val_preprocessed


def split_preprocessed_behavior_data(
    preprocessed_data: PreprocessedTripletEncoderData[T],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[PreprocessedTripletEncoderData[T], PreprocessedTripletEncoderData[T]]:
    """
    Split preprocessed triplet encoder data into train and validation sets.
    
    Args:
        preprocessed_data: Preprocessed triplet encoder data to split
        val_fraction: Fraction of data to use for validation (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (train_preprocessed, val_preprocessed)
    """
    if val_fraction == 0:
        return preprocessed_data, None
    
    n_total = len(preprocessed_data.triplets)
    train_indices, val_indices = _compute_split_indices(n_total, val_fraction, seed)
    
    train_triplets = [preprocessed_data.triplets[i] for i in train_indices]
    val_triplets = [preprocessed_data.triplets[i] for i in val_indices]
    
    train_preprocessed = PreprocessedTripletEncoderData(triplets=train_triplets)
    val_preprocessed = PreprocessedTripletEncoderData(triplets=val_triplets)
    
    return train_preprocessed, val_preprocessed


def split_attention_embedding_preprocessed_data(
    preprocessed_data: PreprocessedAttentionEmbeddingData,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[PreprocessedAttentionEmbeddingData, PreprocessedAttentionEmbeddingData | None]:
    """
    Split preprocessed attention embedding data into train and validation sets.
    
    Splits each model set into train and validation sets.
    
    Args:
        preprocessed_data: Preprocessed attention embedding data to split
        val_fraction: Fraction of pairs to use for validation (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (train_preprocessed, val_preprocessed) with shared scaler.
        val_preprocessed is None if val_fraction is 0.
    """
    if val_fraction == 0:
        return preprocessed_data, None

    train_samples = []
    val_samples = []
    for model_set in preprocessed_data.samples:
        train_indices, val_indices = _compute_split_indices(len(model_set.pairs), val_fraction, seed)
        
        train_model_set = [model_set.pairs[i] for i in train_indices]
        val_model_set = [model_set.pairs[i] for i in val_indices]
        
        train_indexes = [model_set.indexes[i] for i in train_indices]
        val_indexes = [model_set.indexes[i] for i in val_indices]
        
        train_samples.append(ModelSetSample(train_model_set, model_set.model_id, train_indexes))
        val_samples.append(ModelSetSample(val_model_set, model_set.model_id, val_indexes))
    
    train_preprocessed = PreprocessedAttentionEmbeddingData(
        samples=train_samples,
        model_id_to_index=preprocessed_data.model_id_to_index,
        scaler_state=preprocessed_data.scaler_state,
    )
    
    val_preprocessed = PreprocessedAttentionEmbeddingData(
        samples=val_samples,
        model_id_to_index=preprocessed_data.model_id_to_index,
        scaler_state=preprocessed_data.scaler_state,
    )
    
    return train_preprocessed, val_preprocessed


def downsample(data: TrainingData, max_samples: int, seed: int) -> TrainingData:
    """
    Downsample training data to a maximum number of samples.
    
    Args:
        data: Training data to downsample
        max_samples: Maximum number of samples to downsample to
        seed: Random seed for reproducibility
    """
    downsampled_entries = np.random.RandomState(seed).choice(data.entries, size=max_samples, replace=False)
    return TrainingData(entries=downsampled_entries.tolist())
