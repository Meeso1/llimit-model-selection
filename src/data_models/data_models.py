from dataclasses import dataclass
from typing import Literal


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
    # TODO: Add input data here
    pass


@dataclass
class OutputData:
    # TODO: Add output data here
    pass
