"""Data types for simple scoring model."""

from dataclasses import dataclass
from typing import Literal

from src.utils.string_encoder import StringEncoder


@dataclass
class PreprocessedComparison:
    """A single comparison between two models."""
    model_id_a: int
    model_id_b: int
    comparison_type: Literal["model_a_wins", "model_b_wins", "tie", "both_bad"]


@dataclass
class PreprocessedTrainingData:
    """Training data after preprocessing - contains model IDs and encoder."""
    comparisons: list[PreprocessedComparison]
    model_encoder: StringEncoder

