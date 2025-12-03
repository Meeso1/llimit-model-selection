from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field

from src.models.optimizers.optimizer_spec_union import OptimizerSpec


ModelType = Literal["dense_network"]


class ModelSpecBase(BaseModel):
    model_type: ModelType


class DenseNetworkSpecification(ModelSpecBase):
    model_type: Literal["dense_network"] = "dense_network"
    embedding_model_name: str
    hidden_dims: list[int]
    model_id_embedding_dim: int
    optimizer: OptimizerSpec


ModelSpec = Annotated[Union[DenseNetworkSpecification], Field(discriminator="model_type")]
