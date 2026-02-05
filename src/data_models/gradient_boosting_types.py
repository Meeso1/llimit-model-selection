from dataclasses import dataclass
import numpy as np

from src.data_models.data_models import OutputData
from src.utils.string_encoder import StringEncoder


@dataclass
class PreprocessedPromptPair:
    """A pair of models compared for a prompt with embeddings and winner label."""
    prompt_embedding: np.ndarray  # [prompt_embedding_dim]
    prompt_features: np.ndarray  # [prompt_features_dim]
    prompt_categories: np.ndarray  # [prompt_categories_dim]
    model_embedding_a: np.ndarray  # [model_embedding_dim]
    model_embedding_b: np.ndarray  # [model_embedding_dim]
    model_id_a: int
    model_id_b: int
    winner_label: int  # 0 for model_a wins, 1 for model_b wins


@dataclass
class PreprocessedTrainingData:
    """Training data after preprocessing - contains prompt embeddings, model IDs, and encoder."""
    pairs: list[PreprocessedPromptPair]
    embedding_dim: int
    prompt_features_dim: int
    prompt_categories_dim: int
    model_encoder: StringEncoder
    filtered_indexes: list[int]


@dataclass
class PreprocessedInferenceInput:
    """Preprocessed input for inference - prompt embeddings and model IDs."""
    prompt_embeddings: np.ndarray  # [n_prompts, prompt_embedding_dim]
    prompt_features: np.ndarray  # [n_prompts, prompt_features_dim]
    prompt_categories: np.ndarray  # [n_prompts, prompt_categories_dim]
    model_ids: list[int]  # [n_models]


@dataclass
class PromptRoutingOutput(OutputData):
    """Output of dense network model prediction."""
    _scores: dict[str, np.ndarray]  # model_name -> scores array [n_prompts]
    
    @property
    def scores(self) -> dict[str, np.ndarray]:
        """Get scores for each model."""
        return self._scores