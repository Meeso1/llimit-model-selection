from abc import ABC
from dataclasses import dataclass
from typing import Literal


ModelType = Literal["dense_network"]


@dataclass
class OptimizerSpecification:
    optimizer_type: str # TODO: Use `Literal` to validate
    learning_rate: float
    learning_rate_decay: float


@dataclass
class ModelSpecBase(ABC):
    model_type: ModelType


@dataclass
class DenseNetworkSpecification(ModelSpecBase):
    # TODO: Is it possible to restrict `model_type` here?
    embedding_model_name: str
    hidden_dims: list[int]
    model_id_embedding_dim: int
    optimizer: OptimizerSpecification
