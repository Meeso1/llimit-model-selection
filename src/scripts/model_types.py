from typing import Literal
from pydantic import BaseModel

from src.models.optimizers.optimizer_spec import OptimizerSpecification


ModelType = Literal["dense_network"]


class ModelSpecBase(BaseModel):
    model_type: ModelType


class DenseNetworkSpecification(ModelSpecBase):
    model_type: Literal["dense_network"] = "dense_network"
    embedding_model_name: str
    hidden_dims: list[int]
    model_id_embedding_dim: int
    optimizer: OptimizerSpecification
