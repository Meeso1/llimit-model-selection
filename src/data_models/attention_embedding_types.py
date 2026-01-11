from dataclasses import dataclass
import numpy as np


@dataclass
class ProcessedPair:
    """A single processed (prompt, response) pair with all extracted features."""
    prompt_emb: np.ndarray  # [d_emb]
    response_emb: np.ndarray  # [d_emb]
    scalar_features: np.ndarray  # [n_scalar_features]
    model_id: str


@dataclass
class ModelSetSample:
    """A set of pairs from a single model."""
    pairs: list[ProcessedPair]
    model_id: str
    indexes: list[int]


@dataclass
class ScalerState:
    """State of the feature scaler."""
    mean: np.ndarray  # [n_features]
    scale: np.ndarray  # [n_features]


@dataclass
class PreprocessedAttentionEmbeddingData:
    """Preprocessed data for the attention-based embedding model."""
    samples: list[ModelSetSample]
    model_id_to_index: dict[str, int]  # Mapping from model ID to integer index
    scaler_state: ScalerState  # Fitted scaler for inference
