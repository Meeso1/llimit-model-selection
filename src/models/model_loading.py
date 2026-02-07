"""Model loading utilities."""

from typing import Any, Literal

from src.models.model_base import ModelBase
from src.models.length_prediction.length_prediction_model_base import LengthPredictionModelBase
from src.models.scoring_model_base import ScoringModelBase
from src.models.has_embedding_model import HasEmbeddingModel
from src.models.embedding_models.embedding_model_base import EmbeddingModelBase


ScoringModelType = Literal[
    "dense_network",
    "dn_embedding",
    "simple_scoring",
    "elo_scoring",
    "greedy_ranking",
    "mcmf_scoring",
    "least_squares_scoring",
    "gradient_boosting",
    "transformer_embedding",
]

LengthPredictionModelType = Literal[
    "dn_embedding_length_prediction",
]

ModelType = ScoringModelType | LengthPredictionModelType


def load_model(model_type: ModelType, model_name: str) -> ModelBase:
    """
    Load a model by type and name.
    
    Args:
        model_type: Type of the model to load
        model_name: Name of the saved model
        
    Returns:
        Loaded model instance
    """
    match model_type:
        case "dense_network":
            from src.models.dense_network_model import DenseNetworkModel
            return DenseNetworkModel.load(model_name)
        case "dn_embedding":
            from src.models.dn_embedding_model import DnEmbeddingModel
            return DnEmbeddingModel.load(model_name)
        case "simple_scoring":
            from src.models.simple_scoring_model import SimpleScoringModel
            return SimpleScoringModel.load(model_name)
        case "elo_scoring":
            from src.models.elo_scoring_model import EloScoringModel
            return EloScoringModel.load(model_name)
        case "greedy_ranking":
            from src.models.greedy_ranking_model import GreedyRankingModel
            return GreedyRankingModel.load(model_name)
        case "mcmf_scoring":
            from src.models.mcmf_scoring_model import McmfScoringModel
            return McmfScoringModel.load(model_name)
        case "least_squares_scoring":
            from src.models.least_squares_scoring_model import LeastSquaresScoringModel
            return LeastSquaresScoringModel.load(model_name)
        case "gradient_boosting":
            from src.models.gradient_boosting_model import GradientBoostingModel
            return GradientBoostingModel.load(model_name)
        case "transformer_embedding":
            from src.models.transformer_embedding_model import TransformerEmbeddingModel
            return TransformerEmbeddingModel.load(model_name)
        case "dn_embedding_length_prediction":
            from src.models.length_prediction.dn_embedding_length_prediction_model import DnEmbeddingLengthPredictionModel
            return DnEmbeddingLengthPredictionModel.load(model_name)
        case unknown:
            raise ValueError(f"Unknown model type: {unknown}")  # pyright: ignore[reportUnreachable]


def load_scoring_model(model_type: ScoringModelType, model_name: str) -> ScoringModelBase:
    return ScoringModelBase.assert_kind(load_model(model_type, model_name))


def load_length_prediction_model(model_type: LengthPredictionModelType, model_name: str) -> LengthPredictionModelBase:
    return LengthPredictionModelBase.assert_kind(load_model(model_type, model_name))


def load_model_from_state_dict(model_type: ModelType, state_dict: dict[str, Any]) -> ModelBase:
    """
    Load a model from a state dict.
    
    Args:
        model_type: Type of the model to load
        state_dict: State dict from get_state_dict()
        
    Returns:
        Loaded model instance
    """
    match model_type:
        case "dense_network":
            from src.models.dense_network_model import DenseNetworkModel
            return DenseNetworkModel.load_state_dict(state_dict)
        case "dn_embedding":
            from src.models.dn_embedding_model import DnEmbeddingModel
            return DnEmbeddingModel.load_state_dict(state_dict)
        case "simple_scoring":
            from src.models.simple_scoring_model import SimpleScoringModel
            return SimpleScoringModel.load_state_dict(state_dict)
        case "elo_scoring":
            from src.models.elo_scoring_model import EloScoringModel
            return EloScoringModel.load_state_dict(state_dict)
        case "greedy_ranking":
            from src.models.greedy_ranking_model import GreedyRankingModel
            return GreedyRankingModel.load_state_dict(state_dict)
        case "mcmf_scoring":
            from src.models.mcmf_scoring_model import McmfScoringModel
            return McmfScoringModel.load_state_dict(state_dict)
        case "least_squares_scoring":
            from src.models.least_squares_scoring_model import LeastSquaresScoringModel
            return LeastSquaresScoringModel.load_state_dict(state_dict)
        case "gradient_boosting":
            from src.models.gradient_boosting_model import GradientBoostingModel
            return GradientBoostingModel.load_state_dict(state_dict)
        case "transformer_embedding":
            from src.models.transformer_embedding_model import TransformerEmbeddingModel
            return TransformerEmbeddingModel.load_state_dict(state_dict)
        case "dn_embedding_length_prediction":
            from src.models.length_prediction.dn_embedding_length_prediction_model import DnEmbeddingLengthPredictionModel
            return DnEmbeddingLengthPredictionModel.load_state_dict(state_dict)
        case unknown:
            raise ValueError(f"Unknown model type: {unknown}")  # pyright: ignore[reportUnreachable]


def load_scoring_model_from_state_dict(model_type: ScoringModelType, state_dict: dict[str, Any]) -> ScoringModelBase:
    return ScoringModelBase.assert_kind(load_model_from_state_dict(model_type, state_dict))


def load_embedding_model_from_model(model_spec: str) -> EmbeddingModelBase:
    """
    Load an embedding model from a saved model that contains one.
    
    Args:
        model_spec: Model specification in the format "model_type/model_name"
        
    Returns:
        The extracted embedding model
        
    Raises:
        ValueError: If the model_spec format is invalid
        RuntimeError: If the loaded model doesn't have an embedding model or it's not initialized
    """
    if "/" not in model_spec:
        raise ValueError(
            f"model_spec must be in the format 'model_type/model_name', got: {model_spec}"
        )
    
    model_type, model_name = model_spec.split("/", 1)
    
    # Load the model using the standard loading mechanism
    loaded_model = load_model(model_type, model_name)  # type: ignore[arg-type]
    
    # Check if the model has an embedding model
    if not isinstance(loaded_model, HasEmbeddingModel):
        raise RuntimeError(
            f"Model {model_spec} does not contain an embedding model. "
            f"Only models implementing HasEmbeddingModel protocol can be used."
        )
    
    # Extract the embedding model
    embedding_model = loaded_model.embedding_model
    
    if embedding_model is None:
        raise RuntimeError(
            f"Model {model_spec} has no initialized embedding model"
        )
    
    return embedding_model
