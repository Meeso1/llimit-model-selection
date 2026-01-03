from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field

from src.models.optimizers.optimizer_spec_union import OptimizerSpec
from src.models.embedding_specs.embedding_spec_union import EmbeddingSpec
from src.models.finetuning_specs.finetuning_spec_union import FineTuningSpec


ModelType = Literal["dense_network", "dn_embedding", "simple_scoring", "elo_scoring", "greedy_ranking", "mcmf_scoring", "least_squares_scoring", "gradient_boosting", "transformer_embedding"]


class ModelSpecBase(BaseModel):
    model_type: ModelType


class DenseNetworkSpecification(ModelSpecBase):
    model_type: Literal["dense_network"] = "dense_network"
    embedding_model_name: str
    hidden_dims: list[int]
    model_id_embedding_dim: int
    optimizer: OptimizerSpec


class DnEmbeddingSpecification(ModelSpecBase):
    model_type: Literal["dn_embedding"] = "dn_embedding"
    hidden_dims: list[int]
    optimizer: OptimizerSpec
    balance_model_samples: bool = True
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_spec: EmbeddingSpec
    min_model_comparisons: int = 20
    embedding_model_epochs: int = 10


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


class McmfScoringSpecification(ModelSpecBase):
    model_type: Literal["mcmf_scoring"] = "mcmf_scoring"
    min_model_occurrences: int = 1000
    print_summary: bool = True


class LeastSquaresScoringSpecification(ModelSpecBase):
    model_type: Literal["least_squares_scoring"] = "least_squares_scoring"
    min_model_occurrences: int = 1000
    print_summary: bool = True


class GradientBoostingSpecification(ModelSpecBase):
    model_type: Literal["gradient_boosting"] = "gradient_boosting"
    max_depth: int = 6
    learning_rate: float = 0.1
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    balance_model_samples: bool = True
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_spec: EmbeddingSpec
    min_model_comparisons: int = 20
    embedding_model_epochs: int = 10


class TransformerEmbeddingSpecification(ModelSpecBase):
    model_type: Literal["transformer_embedding"] = "transformer_embedding"
    transformer_model_name: str = "sentence-transformers/all-MiniLM-L12-v2"
    finetuning_spec: FineTuningSpec
    hidden_dims: list[int]
    dropout: float = 0.2
    max_length: int = 256
    optimizer: OptimizerSpec
    balance_model_samples: bool = True
    embedding_spec: EmbeddingSpec
    load_embedding_model_from: str | None = None
    min_model_comparisons: int = 20
    embedding_model_epochs: int = 10
    seed: int = 42


ModelSpec = Annotated[
    Union[DenseNetworkSpecification, DnEmbeddingSpecification, SimpleScoringSpecification, EloScoringSpecification, GreedyRankingSpecification, McmfScoringSpecification, LeastSquaresScoringSpecification, GradientBoostingSpecification, TransformerEmbeddingSpecification], 
    Field(discriminator="model_type")
]
