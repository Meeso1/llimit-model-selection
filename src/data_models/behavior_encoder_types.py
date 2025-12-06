from dataclasses import dataclass
import numpy as np


@dataclass
class PromptResponsePair:
    """A single (prompt, response) pair."""
    prompt: str
    response: str


@dataclass
class TrainingTriplet:
    """A single training triplet for the ModelBehaviorEncoder."""
    anchor_prompt: str
    anchor_response: str
    
    positive_prompt: str
    positive_response: str
    
    negative_prompt: str
    negative_response: str


class TripletEmbedding:
    """Embeddings for a single training triplet."""
    anchor_prompt: np.ndarray # [embedding_dim]
    anchor_response: np.ndarray  # [embedding_dim]
    positive_prompt: np.ndarray  # [embedding_dim]
    positive_response: np.ndarray  # [embedding_dim]
    negative_prompt: np.ndarray  # [embedding_dim]
    negative_response: np.ndarray  # [embedding_dim]


class PromptResponsePairEmbedding:
    """Embeddings for a single prompt-response pair."""
    prompt: np.ndarray # [embedding_dim]
    response: np.ndarray  # [embedding_dim]


# TODO: Update usages
@dataclass
class PreprocessedBehaviorEncoderData:
    """Preprocessed data for the ModelBehaviorEncoder."""
    triplets: list[TripletEmbedding]
    # Other preprocessed fields can be added here.


@dataclass
class ModelBehaviorEncoderOutput:
    """Output from the ModelBehaviorEncoder."""
    model_embeddings: dict[str, np.ndarray]
