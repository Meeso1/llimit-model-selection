"""Data models for response length prediction."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from src.utils.string_encoder import StringEncoder


@dataclass
class LengthPredictionOutputData:
    """Output data from length prediction models."""
    predictions: dict[str, np.ndarray]  # model_name -> predicted lengths array [n_prompts]


@dataclass
class PreprocessedLengthPredictionSample:
    """Preprocessed single sample for length prediction (without model embedding)."""
    prompt_embedding: np.ndarray  # [prompt_embedding_dim]
    prompt_features: np.ndarray  # [prompt_features_dim]
    model_id_a: int
    model_id_b: int
    log_response_length_a: float  # Scaled response length for model A
    log_response_length_b: float  # Scaled response length for model B


@dataclass
class PreprocessedLengthPredictionSampleWithEmbedding:
    """Preprocessed single sample for length prediction (with model embedding)."""
    prompt_embedding: np.ndarray  # [prompt_embedding_dim]
    prompt_features: np.ndarray  # [prompt_features_dim]
    model_embedding_a: np.ndarray  # [model_embedding_dim]
    model_embedding_b: np.ndarray  # [model_embedding_dim]
    model_id_a: int
    model_id_b: int
    log_response_length_a: float  # Scaled response length for model A
    log_response_length_b: float  # Scaled response length for model B


@dataclass
class PreprocessedLengthPredictionTrainingData:
    """Training data after preprocessing for length prediction (without model embeddings)."""
    samples: list[PreprocessedLengthPredictionSample]
    embedding_dim: int
    prompt_features_dim: int
    model_encoder: StringEncoder
    filtered_indexes: list[int]
    output_scaler_state: dict[str, Any]
    prompt_features_scaler_state: dict[str, Any]
    
    def add_model_embeddings(
        self, 
        model_embeddings: dict[str, np.ndarray],
        model_embedding_dim: int,
    ) -> "PreprocessedLengthPredictionTrainingDataWithEmbeddings":
        """
        Add model embeddings to create the full preprocessed data.
        
        Args:
            model_embeddings: Dictionary mapping model names to embeddings [model_embedding_dim]
            model_embedding_dim: Dimension of model embeddings
            
        Returns:
            PreprocessedLengthPredictionTrainingDataWithEmbeddings with model embeddings added
        """
        samples_with_embeddings = []
        for sample in self.samples:
            model_name_a = self.model_encoder.decode(sample.model_id_a)
            model_name_b = self.model_encoder.decode(sample.model_id_b)
            
            samples_with_embeddings.append(
                PreprocessedLengthPredictionSampleWithEmbedding(
                    prompt_embedding=sample.prompt_embedding,
                    prompt_features=sample.prompt_features,
                    model_embedding_a=model_embeddings[model_name_a],
                    model_embedding_b=model_embeddings[model_name_b],
                    model_id_a=sample.model_id_a,
                    model_id_b=sample.model_id_b,
                    log_response_length_a=sample.log_response_length_a,
                    log_response_length_b=sample.log_response_length_b,
                )
            )
        
        return PreprocessedLengthPredictionTrainingDataWithEmbeddings(
            samples=samples_with_embeddings,
            embedding_dim=self.embedding_dim,
            prompt_features_dim=self.prompt_features_dim,
            model_embedding_dim=model_embedding_dim,
            model_encoder=self.model_encoder,
            filtered_indexes=self.filtered_indexes,
            output_scaler_state=self.output_scaler_state,
            prompt_features_scaler_state=self.prompt_features_scaler_state,
        )


@dataclass
class PreprocessedLengthPredictionTrainingDataWithEmbeddings:
    """Training data after preprocessing for length prediction (with model embeddings)."""
    samples: list[PreprocessedLengthPredictionSampleWithEmbedding]
    embedding_dim: int
    prompt_features_dim: int
    model_embedding_dim: int
    model_encoder: StringEncoder
    filtered_indexes: list[int]
    output_scaler_state: dict[str, Any]
    prompt_features_scaler_state: dict[str, Any]


@dataclass
class PreprocessedLengthPredictionInferenceInput:
    """Preprocessed input for length prediction inference."""
    prompt_embeddings: np.ndarray  # [n_prompts, prompt_embedding_dim]
    prompt_features: np.ndarray  # [n_prompts, prompt_features_dim]
