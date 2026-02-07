from abc import ABC
from src.models.model_base import ModelBase, ModelKind
from src.data_models.data_models import OutputData


class ScoringModelBase(ModelBase[OutputData], ABC):
    """Base class for scoring models."""
    
    @property
    def model_kind(self) -> ModelKind:
        """Return the kind of model."""
        return "scoring"

    @staticmethod
    def assert_kind(model: ModelBase) -> "ScoringModelBase":
        if model.model_kind != "scoring":
            raise ValueError(f"Expected scoring model, but got {model.model_kind} model")

        return model
