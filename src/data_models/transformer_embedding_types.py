from dataclasses import dataclass
from typing import Any
import numpy as np

from src.data_models.data_models import OutputData
from src.utils.string_encoder import StringEncoder


@dataclass
class PreprocessedPromptPair:
    """A pair of models compared for a prompt with raw text and winner label."""
    prompt: str  # Raw prompt text
    prompt_features: np.ndarray  # [prompt_features_dim]
    model_embedding_a: np.ndarray  # [model_embedding_dim]
    model_embedding_b: np.ndarray  # [model_embedding_dim]
    model_id_a: int
    model_id_b: int
    winner_label: int  # 0 for model_a wins, 1 for model_b wins


@dataclass
class PreprocessedTrainingData:
    """Training data after preprocessing - contains raw prompts, model IDs, and encoder."""
    pairs: list[PreprocessedPromptPair]
    prompt_features_dim: int
    model_encoder: StringEncoder
    filtered_indexes: list[int]
    scaler_state: dict[str, Any]


@dataclass
class PreprocessedInferenceInput:
    """Preprocessed input for inference - raw prompts and model embeddings."""
    prompts: list[str]  # [n_prompts]
    prompt_features: np.ndarray  # [n_prompts, prompt_features_dim]
    model_embeddings: np.ndarray  # [n_models, model_embedding_dim]


@dataclass
class PromptRoutingOutput(OutputData):
    """Output of transformer embedding model prediction."""
    _scores: dict[str, np.ndarray]  # model_name -> scores array [n_prompts]
    
    @property
    def scores(self) -> dict[str, np.ndarray]:
        """Get scores for each model."""
        return self._scores

