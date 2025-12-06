from dataclasses import dataclass
import numpy as np


@dataclass
class PromptResponsePair:
    """A single (prompt, response) pair."""
    prompt: str
    response: str


@dataclass
class BehaviorEncoderTrainingTriplet:
    """A single training triplet for the ModelBehaviorEncoder."""
    anchor_prompt: str
    anchor_response: str
    
    positive_prompt: str
    positive_response: str
    
    negative_prompt: str
    negative_response: str


@dataclass
class PreprocessedBehaviorEncoderData:
    """Preprocessed data for the ModelBehaviorEncoder."""
    triplets: list[BehaviorEncoderTrainingTriplet]
    # Other preprocessed fields can be added here.


@dataclass
class ModelBehaviorEncoderOutput:
    """Output from the ModelBehaviorEncoder."""
    model_embeddings: dict[str, np.ndarray]
