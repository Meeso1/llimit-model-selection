from dataclasses import dataclass
from typing import TypeVar, Generic
import numpy as np


@dataclass
class PromptResponsePair:
    """A single (prompt, response) pair."""
    prompt: str
    response: str


@dataclass
class TrainingTriplet:
    """A single training triplet for triplet-based encoder models (text-based)."""
    anchor_prompt: str
    anchor_response: str
    anchor_model_id: str
    
    positive_prompt: str
    positive_response: str
    
    negative_prompt: str
    negative_response: str


@dataclass
class TripletEmbedding:
    """Embeddings for a single training triplet."""
    anchor_prompt: np.ndarray  # [embedding_dim]
    anchor_response: np.ndarray  # [embedding_dim]
    anchor_model_id: str
    positive_prompt: np.ndarray  # [embedding_dim]
    positive_response: np.ndarray  # [embedding_dim]
    negative_prompt: np.ndarray  # [embedding_dim]
    negative_response: np.ndarray  # [embedding_dim]


@dataclass
class PromptResponsePairEmbedding:
    """Embeddings for a single prompt-response pair."""
    prompt: np.ndarray  # [embedding_dim]
    response: np.ndarray  # [embedding_dim]


# Type variable for the triplet type (either TripletEmbedding or TrainingTriplet)
TripletType = TypeVar('TripletType', TripletEmbedding, TrainingTriplet)


@dataclass
class PreprocessedTripletEncoderData(Generic[TripletType]):
    """
    Preprocessed data for triplet-based encoder models.
    
    Generic over the triplet type:
    - PreprocessedTripletEncoderData[TripletEmbedding] for frozen encoder (pre-computed embeddings)
    - PreprocessedTripletEncoderData[TrainingTriplet] for fine-tunable encoder (raw text)
    """
    triplets: list[TripletType]


@dataclass
class TripletEncoderOutput:
    """Output from triplet-based encoder models."""
    model_embeddings: dict[str, np.ndarray]

