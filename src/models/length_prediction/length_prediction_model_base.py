"""Base class for length prediction models."""

from abc import ABC
from src.models.model_base import ModelBase, ModelKind
from src.data_models.length_prediction.length_prediction_data_models import LengthPredictionOutputData


class LengthPredictionModelBase(ModelBase[LengthPredictionOutputData], ABC):
    """Base class for all length prediction models."""
    
    @property
    def model_kind(self) -> ModelKind:
        """Return the kind of model."""
        return "length_prediction"

    @staticmethod
    def assert_kind(model: ModelBase) -> "LengthPredictionModelBase":
        if model.model_kind != "length_prediction":
            raise ValueError(f"Expected length prediction model, but got {model.model_kind} model")

        return model
