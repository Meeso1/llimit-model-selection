from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field

from src.models.optimizers.optimizer_spec_union import OptimizerSpec


ModelType = Literal["dense_network", "simple_scoring", "elo_scoring", "greedy_ranking"]


class ModelSpecBase(BaseModel):
    model_type: ModelType


class DenseNetworkSpecification(ModelSpecBase):
    model_type: Literal["dense_network"] = "dense_network"
    embedding_model_name: str
    hidden_dims: list[int]
    model_id_embedding_dim: int
    optimizer: OptimizerSpec


class SimpleScoringSpecification(ModelSpecBase):
    model_type: Literal["simple_scoring"] = "simple_scoring"
    optimizer: OptimizerSpec
    balance_model_samples: bool = True
    tie_both_bad_epsilon: float = 1e-2
    non_ranking_loss_coeff: float = 0.01
    min_model_occurrences: int = 1000


class EloScoringSpecification(ModelSpecBase):
    model_type: Literal["elo_scoring"] = "elo_scoring"
    initial_rating: float = 1500.0
    k_factor: float = 32.0
    balance_model_samples: bool = True
    tie_both_bad_epsilon: float = 100.0
    non_ranking_loss_coeff: float = 0.1
    min_model_occurrences: int = 1000


class GreedyRankingSpecification(ModelSpecBase):
    model_type: Literal["greedy_ranking"] = "greedy_ranking"
    min_model_occurrences: int = 1000
    score_normalization: str = "negative_rank"
    print_summary: bool = True


ModelSpec = Annotated[
    Union[DenseNetworkSpecification, SimpleScoringSpecification, EloScoringSpecification, GreedyRankingSpecification], 
    Field(discriminator="model_type")
]
