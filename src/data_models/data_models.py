from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass
class EvaluationMessage:
    role: Literal["user", "assistant", "system"]
    content: str


@dataclass
class CreativeWritingTag:
    creative_writing: bool
    score: str


@dataclass
class CriteriaTag:
    complexity: bool
    creativity: bool
    domain_knowledge: bool
    problem_solving: bool
    real_world: bool
    specificity: bool
    technical_accuracy: bool


@dataclass
class IfTag:
    if_: bool  # 'if' is a Python keyword, so we use 'if_'
    score: int


@dataclass
class MathTag:
    math: bool


@dataclass
class CategoryTag:
    creative_writing_v0_1: CreativeWritingTag
    criteria_v0_1: CriteriaTag
    if_v0_1: IfTag
    math_v0_1: MathTag


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
    category_tag: CategoryTag | None = None


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
