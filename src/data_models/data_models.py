from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass
class EvaluationMessage:
    role: Literal["user", "assistant", "system"]
    content: str


@dataclass
class EvaluationEntry:
    model_a: str
    model_b: str
    winner: Literal["model_a", "model_b", "tie", "both_bad"]
    evaluation_session_id: str
    evaluation_order: int
    conversation_history: list[EvaluationMessage]
    user_prompt: str
    model_a_response: str
    model_b_response: str
    timestamp: str


@dataclass
class TrainingData:
    entries: list[EvaluationEntry]


@dataclass
class InputData:
    """Model-invariant input data for inference."""
    prompts: list[str]
    model_names: list[str]


class OutputData(ABC):
    """Abstract base class for model output data."""
    
    @property
    @abstractmethod
    def scores(self) -> dict[str, np.ndarray]:
        """
        Get scores for each model.
        
        Returns:
            Dictionary mapping model names to score arrays [n_prompts]
        """
        pass
