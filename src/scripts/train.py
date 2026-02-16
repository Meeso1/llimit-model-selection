import sys
from typing import Any
import datasets
from pathlib import Path

from src import data_loading
from src.data_models.data_models import TrainingData
from src.models.scoring.dense_network_model import DenseNetworkModel
from src.models.scoring.dn_embedding_model import DnEmbeddingModel
from src.models.scoring.simple_scoring_model import SimpleScoringModel
from src.models.scoring.elo_scoring_model import EloScoringModel
from src.models.scoring.greedy_ranking_model import GreedyRankingModel
from src.models.scoring.mcmf_scoring_model import McmfScoringModel
from src.models.scoring.least_squares_scoring_model import LeastSquaresScoringModel
from src.models.scoring.gradient_boosting_model import GradientBoostingModel
from src.models.scoring.transformer_embedding_model import TransformerEmbeddingModel
from src.models.scoring.response_predictive_model import ResponsePredictiveModel
from src.models.length_prediction.dn_embedding_length_prediction_model import DnEmbeddingLengthPredictionModel
from src.models.length_prediction.gb_length_prediction_model import GbLengthPredictionModel
from src.models.model_base import ModelBase
from src.scripts.model_types import (
    DenseNetworkSpecification,
    DnEmbeddingSpecification,
    SimpleScoringSpecification,
    EloScoringSpecification,
    GreedyRankingSpecification,
    McmfScoringSpecification,
    LeastSquaresScoringSpecification,
    GradientBoostingSpecification,
    TransformerEmbeddingSpecification,
    ResponsePredictiveSpecification,
    DnEmbeddingLengthPredictionSpecification,
    GbLengthPredictionSpecification,
)
from src.scripts.training_spec import TrainingSpecification
from src.utils import data_split
from src.utils.jars import Jars


def run_train(args: Any) -> None:
    """
    Run training from command line arguments.
    
    Args:
        args: Parsed command line arguments with 'spec_file' attribute
    """
    spec_file: str | None = args.spec_file
    
    if spec_file is not None:
        with open(spec_file, "r") as f:
            spec = TrainingSpecification.model_validate_json(f.read())
    else:
        stdin = sys.stdin.read()
        spec = TrainingSpecification.model_validate_json(stdin)
    
    train(spec)


def train(spec: TrainingSpecification) -> None:
    if spec.jar_base_path is not None:
        Jars.set_base_path(Path(spec.jar_base_path))

    model = _create_starting_model(spec)

    training_data = _load_training_data(spec.data.dataset)
    if spec.data.max_samples is not None:
        downsampled = data_split.downsample(training_data, spec.data.max_samples, spec.data.seed)
        print(f"Downsampled data size: {len(downsampled.entries)}")
    else:
        downsampled = training_data

    model.train(
        downsampled, 
        validation_split=data_split.ValidationSplit(
            val_fraction=spec.data.validation_split, 
            seed=spec.data.seed,
        ),
        epochs=spec.epochs, 
        batch_size=spec.batch_size,
    )

    model.save(spec.model.name)

    print(f"Model saved with name: {spec.model.name}")


def _create_starting_model(spec: TrainingSpecification) -> ModelBase:
    if spec.model.spec is None:
        raise ValueError("Either start_state or spec must be provided in model specification")
    
    match spec.model.spec.model_type:
        case "dense_network":
            return _create_starting_dense_network(spec)
        case "dn_embedding":
            return _create_starting_dn_embedding(spec)
        case "simple_scoring":
            return _create_starting_simple_scoring(spec)
        case "elo_scoring":
            return _create_starting_elo_scoring(spec)
        case "greedy_ranking":
            return _create_starting_greedy_ranking(spec)
        case "mcmf_scoring":
            return _create_starting_mcmf_scoring(spec)
        case "least_squares_scoring":
            return _create_starting_least_squares_scoring(spec)
        case "gradient_boosting":
            return _create_starting_gradient_boosting(spec)
        case "transformer_embedding":
            return _create_starting_transformer_embedding(spec)
        case "response_predictive":
            return _create_starting_response_predictive(spec)
        case "dn_embedding_length_prediction":
            return _create_starting_dn_embedding_length_prediction(spec)
        case "gb_length_prediction":
            return _create_starting_gb_length_prediction(spec)
        case unknown:
            raise ValueError(f"Unknown model type: {unknown}")  # pyright: ignore[reportUnreachable]


def _create_starting_dense_network(training_spec: TrainingSpecification) -> DenseNetworkModel:
    if training_spec.model.start_state is not None:
        return DenseNetworkModel.load(training_spec.model.start_state)
    
    if not isinstance(training_spec.model.spec, DenseNetworkSpecification):
        raise ValueError(f"Expected model specification to be of type {DenseNetworkSpecification.__name__}, but found {type(training_spec.model.spec).__name__}")
    
    model_spec = training_spec.model.spec
    return DenseNetworkModel(
        embedding_model_name=model_spec.embedding_model_name,
        hidden_dims=model_spec.hidden_dims,
        model_id_embedding_dim=model_spec.model_id_embedding_dim,
        optimizer_spec=model_spec.optimizer,
        balance_model_samples=True,
        wandb_details=training_spec.wandb.to_wandb_details() if training_spec.wandb is not None else None,
        print_every=training_spec.log.print_every,
    )


def _create_starting_dn_embedding(training_spec: TrainingSpecification) -> DnEmbeddingModel:
    if training_spec.model.start_state is not None:
        return DnEmbeddingModel.load(training_spec.model.start_state)
    
    if not isinstance(training_spec.model.spec, DnEmbeddingSpecification):
        raise ValueError(f"Expected model specification to be of type {DnEmbeddingSpecification.__name__}, but found {type(training_spec.model.spec).__name__}")
    
    model_spec = training_spec.model.spec
    return DnEmbeddingModel(
        hidden_dims=model_spec.hidden_dims,
        optimizer_spec=model_spec.optimizer,
        balance_model_samples=model_spec.balance_model_samples,
        embedding_model_name=model_spec.embedding_model_name,
        embedding_spec=model_spec.embedding_spec,
        min_model_comparisons=model_spec.min_model_comparisons,
        embedding_model_epochs=model_spec.embedding_model_epochs,
        wandb_details=training_spec.wandb.to_wandb_details() if training_spec.wandb is not None else None,
        print_every=training_spec.log.print_every,
    )


def _create_starting_simple_scoring(training_spec: TrainingSpecification) -> SimpleScoringModel:
    if training_spec.model.start_state is not None:
        return SimpleScoringModel.load(training_spec.model.start_state)
    
    if not isinstance(training_spec.model.spec, SimpleScoringSpecification):
        raise ValueError(f"Expected model specification to be of type {SimpleScoringSpecification.__name__}, but found {type(training_spec.model.spec).__name__}")
    
    model_spec = training_spec.model.spec
    return SimpleScoringModel(
        optimizer_spec=model_spec.optimizer,
        balance_model_samples=model_spec.balance_model_samples,
        tie_both_bad_epsilon=model_spec.tie_both_bad_epsilon,
        non_ranking_loss_coeff=model_spec.non_ranking_loss_coeff,
        min_model_occurrences=model_spec.min_model_occurrences,
        wandb_details=training_spec.wandb.to_wandb_details() if training_spec.wandb is not None else None,
        print_every=training_spec.log.print_every,
    )


def _create_starting_elo_scoring(training_spec: TrainingSpecification) -> EloScoringModel:
    if training_spec.model.start_state is not None:
        return EloScoringModel.load(training_spec.model.start_state)
    
    if not isinstance(training_spec.model.spec, EloScoringSpecification):
        raise ValueError(f"Expected model specification to be of type {EloScoringSpecification.__name__}, but found {type(training_spec.model.spec).__name__}")
    
    model_spec = training_spec.model.spec
    return EloScoringModel(
        initial_rating=model_spec.initial_rating,
        k_factor=model_spec.k_factor,
        balance_model_samples=model_spec.balance_model_samples,
        tie_both_bad_epsilon=model_spec.tie_both_bad_epsilon,
        non_ranking_loss_coeff=model_spec.non_ranking_loss_coeff,
        min_model_occurrences=model_spec.min_model_occurrences,
        wandb_details=training_spec.wandb.to_wandb_details() if training_spec.wandb is not None else None,
        print_every=training_spec.log.print_every,
    )


def _create_starting_greedy_ranking(training_spec: TrainingSpecification) -> GreedyRankingModel:
    if training_spec.model.start_state is not None:
        return GreedyRankingModel.load(training_spec.model.start_state)
    
    if not isinstance(training_spec.model.spec, GreedyRankingSpecification):
        raise ValueError(f"Expected model specification to be of type {GreedyRankingSpecification.__name__}, but found {type(training_spec.model.spec).__name__}")
    
    model_spec = training_spec.model.spec
    return GreedyRankingModel(
        min_model_occurrences=model_spec.min_model_occurrences,
        score_normalization=model_spec.score_normalization,
        print_summary=model_spec.print_summary,
        wandb_details=training_spec.wandb.to_wandb_details() if training_spec.wandb is not None else None,
    )


def _create_starting_mcmf_scoring(training_spec: TrainingSpecification) -> McmfScoringModel:
    if training_spec.model.start_state is not None:
        return McmfScoringModel.load(training_spec.model.start_state)
    
    if not isinstance(training_spec.model.spec, McmfScoringSpecification):
        raise ValueError(f"Expected model specification to be of type {McmfScoringSpecification.__name__}, but found {type(training_spec.model.spec).__name__}")
    
    model_spec = training_spec.model.spec
    return McmfScoringModel(
        min_model_occurrences=model_spec.min_model_occurrences,
        print_summary=model_spec.print_summary,
        wandb_details=training_spec.wandb.to_wandb_details() if training_spec.wandb is not None else None,
    )


def _create_starting_least_squares_scoring(training_spec: TrainingSpecification) -> LeastSquaresScoringModel:
    if training_spec.model.start_state is not None:
        return LeastSquaresScoringModel.load(training_spec.model.start_state)
    
    if not isinstance(training_spec.model.spec, LeastSquaresScoringSpecification):
        raise ValueError(f"Expected model specification to be of type {LeastSquaresScoringSpecification.__name__}, but found {type(training_spec.model.spec).__name__}")
    
    model_spec = training_spec.model.spec
    return LeastSquaresScoringModel(
        min_model_occurrences=model_spec.min_model_occurrences,
        print_summary=model_spec.print_summary,
        wandb_details=training_spec.wandb.to_wandb_details() if training_spec.wandb is not None else None,
    )


def _create_starting_gradient_boosting(training_spec: TrainingSpecification) -> GradientBoostingModel:
    if training_spec.model.start_state is not None:
        return GradientBoostingModel.load(training_spec.model.start_state)
    
    if not isinstance(training_spec.model.spec, GradientBoostingSpecification):
        raise ValueError(f"Expected model specification to be of type {GradientBoostingSpecification.__name__}, but found {type(training_spec.model.spec).__name__}")
    
    model_spec = training_spec.model.spec
    return GradientBoostingModel(
        max_depth=model_spec.max_depth,
        learning_rate=model_spec.learning_rate,
        colsample_bytree=model_spec.colsample_bytree,
        reg_alpha=model_spec.reg_alpha,
        reg_lambda=model_spec.reg_lambda,
        balance_model_samples=model_spec.balance_model_samples,
        input_features=model_spec.input_features,
        embedding_model_name=model_spec.embedding_model_name,
        embedding_spec=model_spec.embedding_spec,
        load_embedding_model_from=model_spec.load_embedding_model_from,
        min_model_comparisons=model_spec.min_model_comparisons,
        embedding_model_epochs=model_spec.embedding_model_epochs,
        base_model_name=model_spec.base_model,
        wandb_details=training_spec.wandb.to_wandb_details() if training_spec.wandb is not None else None,
        print_every=training_spec.log.print_every,
        seed=model_spec.seed,
    )


def _create_starting_transformer_embedding(training_spec: TrainingSpecification) -> TransformerEmbeddingModel:
    if training_spec.model.start_state is not None:
        return TransformerEmbeddingModel.load(training_spec.model.start_state)
    
    if not isinstance(training_spec.model.spec, TransformerEmbeddingSpecification):
        raise ValueError(f"Expected model specification to be of type {TransformerEmbeddingSpecification.__name__}, but found {type(training_spec.model.spec).__name__}")
    
    model_spec = training_spec.model.spec
    return TransformerEmbeddingModel(
        transformer_model_name=model_spec.transformer_model_name,
        finetuning_spec=model_spec.finetuning_spec,
        hidden_dims=model_spec.hidden_dims,
        dropout=model_spec.dropout,
        max_length=model_spec.max_length,
        optimizer_spec=model_spec.optimizer,
        balance_model_samples=model_spec.balance_model_samples,
        embedding_spec=model_spec.embedding_spec,
        load_embedding_model_from=model_spec.load_embedding_model_from,
        min_model_comparisons=model_spec.min_model_comparisons,
        embedding_model_epochs=model_spec.embedding_model_epochs,
        scoring_head_lr_multiplier=model_spec.scoring_head_lr_multiplier,
        base_model_name=model_spec.base_model,
        wandb_details=training_spec.wandb.to_wandb_details() if training_spec.wandb is not None else None,
        print_every=training_spec.log.print_every,
        seed=model_spec.seed,
    )


def _create_starting_response_predictive(training_spec: TrainingSpecification) -> ResponsePredictiveModel:
    if training_spec.model.start_state is not None:
        return ResponsePredictiveModel.load(training_spec.model.start_state)
    
    if not isinstance(training_spec.model.spec, ResponsePredictiveSpecification):
        raise ValueError(f"Expected model specification to be of type {ResponsePredictiveSpecification.__name__}, but found {type(training_spec.model.spec).__name__}")
    
    model_spec = training_spec.model.spec
    return ResponsePredictiveModel(
        response_repr_dim=model_spec.response_repr_dim,
        encoder_hidden_dims=model_spec.encoder_hidden_dims,
        prediction_loss_weight=model_spec.prediction_loss_weight,
        predictor_hidden_dims=model_spec.predictor_hidden_dims,
        scorer_hidden_dims=model_spec.scorer_hidden_dims,
        dropout=model_spec.dropout,
        real_repr_ratio=model_spec.real_repr_ratio,
        real_repr_decay_per_epoch=model_spec.real_repr_decay_per_epoch,
        optimizer_spec=model_spec.optimizer,
        balance_model_samples=model_spec.balance_model_samples,
        embedding_model_name=model_spec.embedding_model_name,
        embedding_spec=model_spec.embedding_spec,
        load_embedding_model_from=model_spec.load_embedding_model_from,
        min_model_comparisons=model_spec.min_model_comparisons,
        embedding_model_epochs=model_spec.embedding_model_epochs,
        wandb_details=training_spec.wandb.to_wandb_details() if training_spec.wandb is not None else None,
        print_every=training_spec.log.print_every,
        seed=model_spec.seed,
    )


def _create_starting_dn_embedding_length_prediction(training_spec: TrainingSpecification) -> DnEmbeddingLengthPredictionModel:
    if training_spec.model.start_state is not None:
        return DnEmbeddingLengthPredictionModel.load(training_spec.model.start_state)
    
    if not isinstance(training_spec.model.spec, DnEmbeddingLengthPredictionSpecification):
        raise ValueError(f"Expected model specification to be of type {DnEmbeddingLengthPredictionSpecification.__name__}, but found {type(training_spec.model.spec).__name__}")
    
    model_spec = training_spec.model.spec
    return DnEmbeddingLengthPredictionModel(
        hidden_dims=model_spec.hidden_dims,
        optimizer_spec=model_spec.optimizer,
        embedding_model_name=model_spec.embedding_model_name,
        embedding_spec=model_spec.embedding_spec,
        load_embedding_model_from=model_spec.load_embedding_model_from,
        min_model_comparisons=model_spec.min_model_comparisons,
        embedding_model_epochs=model_spec.embedding_model_epochs,
        wandb_details=training_spec.wandb.to_wandb_details() if training_spec.wandb is not None else None,
        print_every=training_spec.log.print_every,
        seed=model_spec.seed,
    )


def _create_starting_gb_length_prediction(training_spec: TrainingSpecification) -> GbLengthPredictionModel:
    if training_spec.model.start_state is not None:
        return GbLengthPredictionModel.load(training_spec.model.start_state)
    
    if not isinstance(training_spec.model.spec, GbLengthPredictionSpecification):
        raise ValueError(f"Expected model specification to be of type {GbLengthPredictionSpecification.__name__}, but found {type(training_spec.model.spec).__name__}")
    
    model_spec = training_spec.model.spec
    return GbLengthPredictionModel(
        max_depth=model_spec.max_depth,
        learning_rate=model_spec.learning_rate,
        colsample_bytree=model_spec.colsample_bytree,
        colsample_bylevel=model_spec.colsample_bylevel,
        reg_alpha=model_spec.reg_alpha,
        reg_lambda=model_spec.reg_lambda,
        input_features=model_spec.input_features,
        embedding_model_name=model_spec.embedding_model_name,
        embedding_spec=model_spec.embedding_spec,
        load_embedding_model_from=model_spec.load_embedding_model_from,
        min_model_comparisons=model_spec.min_model_comparisons,
        embedding_model_epochs=model_spec.embedding_model_epochs,
        wandb_details=training_spec.wandb.to_wandb_details() if training_spec.wandb is not None else None,
        print_every=training_spec.log.print_every,
        seed=model_spec.seed,
    )


def _load_training_data(dataset_type: str) -> TrainingData:
    """
    Load training data based on the specified dataset type.
    
    Args:
        dataset_type: One of 'lmarena', 'chatbot_arena', or 'both'
    
    Returns:
        TrainingData instance with loaded entries
    """
    match dataset_type:
        case "lmarena_human_preference":
            return _load_lmarena_human_preference()
        case "chatbot_arena":
            return _load_chatbot_arena()
        case "both":
            lmarena_data = _load_lmarena_human_preference()
            chatbot_arena_data = _load_chatbot_arena()
            combined_entries = lmarena_data.entries + chatbot_arena_data.entries
            print(f"Successfully loaded {len(combined_entries)} entries total (lmarena: {len(lmarena_data.entries)}, chatbot_arena: {len(chatbot_arena_data.entries)})")
            return TrainingData(entries=combined_entries)
        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


def _load_lmarena_human_preference() -> TrainingData:
    """Load data from lmarena-ai/arena-human-preference-140k dataset."""
    dataset = datasets.load_dataset("lmarena-ai/arena-human-preference-140k")
    training_data = data_loading.load_training_data_lmarena(dataset["train"].to_pandas())
    print(f"Successfully loaded {len(training_data.entries)} entries from lmarena_human_preference dataset")
    return training_data


def _load_chatbot_arena() -> TrainingData:
    """Load data from lmsys/chatbot_arena_conversations dataset."""
    dataset = datasets.load_dataset("lmsys/chatbot_arena_conversations")
    training_data = data_loading.load_training_data_chatbot_arena(dataset["train"].to_pandas())
    print(f"Successfully loaded {len(training_data.entries)} entries from chatbot_arena dataset")
    return training_data

