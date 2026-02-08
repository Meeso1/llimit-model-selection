"""Data types for response predictive model."""

from dataclasses import dataclass
import numpy as np

from src.data_models.data_models import OutputData
from src.utils.string_encoder import StringEncoder


@dataclass
class ResponsePredictivePair:
    """A pair of models compared with response embeddings and features (without model embeddings)."""
    prompt_embedding: np.ndarray  # [d_prompt_emb]
    prompt_features: np.ndarray  # [d_prompt_features]
    response_embedding_a: np.ndarray  # [d_response_emb]
    response_embedding_b: np.ndarray  # [d_response_emb]
    response_features_a: np.ndarray  # [d_response_features]
    response_features_b: np.ndarray  # [d_response_features]
    model_id_a: int
    model_id_b: int
    winner_label: int  # 0 = model_a wins, 1 = model_b wins


@dataclass
class ResponsePredictivePairWithEmbedding:
    """A pair of models compared with response embeddings and features (with model embeddings)."""
    prompt_embedding: np.ndarray  # [d_prompt_emb]
    prompt_features: np.ndarray  # [d_prompt_features]
    response_embedding_a: np.ndarray  # [d_response_emb]
    response_embedding_b: np.ndarray  # [d_response_emb]
    response_features_a: np.ndarray  # [d_response_features]
    response_features_b: np.ndarray  # [d_response_features]
    model_embedding_a: np.ndarray  # [d_model_emb]
    model_embedding_b: np.ndarray  # [d_model_emb]
    model_id_a: int
    model_id_b: int
    winner_label: int  # 0 = model_a wins, 1 = model_b wins


@dataclass
class PreprocessedTrainingData:
    """Training data after preprocessing (without model embeddings)."""
    pairs: list[ResponsePredictivePair]
    prompt_features_dim: int
    response_features_dim: int
    model_encoder: StringEncoder
    filtered_indexes: list[int]
    prompt_scaler_state: dict
    response_scaler_state: dict

    def add_model_embeddings(
        self,
        model_embeddings: dict[str, np.ndarray],
    ) -> "PreprocessedTrainingDataWithEmbeddings":
        """
        Add model embeddings to create the full preprocessed data.
        
        Args:
            model_embeddings: Dictionary mapping model names to embeddings [model_embedding_dim]
            
        Returns:
            PreprocessedTrainingDataWithEmbeddings with model embeddings added
        """
        pairs_with_embeddings = []
        for pair in self.pairs:
            model_name_a = self.model_encoder.decode(pair.model_id_a)
            model_name_b = self.model_encoder.decode(pair.model_id_b)

            pairs_with_embeddings.append(
                ResponsePredictivePairWithEmbedding(
                    prompt_embedding=pair.prompt_embedding,
                    prompt_features=pair.prompt_features,
                    response_embedding_a=pair.response_embedding_a,
                    response_embedding_b=pair.response_embedding_b,
                    response_features_a=pair.response_features_a,
                    response_features_b=pair.response_features_b,
                    model_embedding_a=model_embeddings[model_name_a],
                    model_embedding_b=model_embeddings[model_name_b],
                    model_id_a=pair.model_id_a,
                    model_id_b=pair.model_id_b,
                    winner_label=pair.winner_label,
                )
            )

        return PreprocessedTrainingDataWithEmbeddings(
            pairs=pairs_with_embeddings,
            prompt_features_dim=self.prompt_features_dim,
            response_features_dim=self.response_features_dim,
            model_encoder=self.model_encoder,
            filtered_indexes=self.filtered_indexes,
            prompt_scaler_state=self.prompt_scaler_state,
            response_scaler_state=self.response_scaler_state,
        )


@dataclass
class PreprocessedTrainingDataWithEmbeddings:
    """Training data after preprocessing (with model embeddings)."""
    pairs: list[ResponsePredictivePairWithEmbedding]
    prompt_features_dim: int
    response_features_dim: int
    model_encoder: StringEncoder
    filtered_indexes: list[int]
    prompt_scaler_state: dict
    response_scaler_state: dict


@dataclass
class ResponsePredictiveInferenceData:
    """Preprocessed input for inference."""
    prompt_embeddings: np.ndarray  # [n_prompts, d_prompt_emb]
    prompt_features: np.ndarray  # [n_prompts, d_prompt_features]
    model_embeddings: np.ndarray  # [n_models, d_model_emb]


@dataclass
class PromptRoutingOutput(OutputData):
    """Output of response predictive model prediction."""
    _scores: dict[str, np.ndarray]  # model_name -> scores array [n_prompts]

    @property
    def scores(self) -> dict[str, np.ndarray]:
        """Get scores for each model."""
        return self._scores
