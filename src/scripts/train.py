import sys
from typing import Any
import datasets

from src import data_loading
from src.data_models.data_models import TrainingData
from src.models.dense_network_model import DenseNetworkModel
from src.models.simple_scoring_model import SimpleScoringModel
from src.models.elo_scoring_model import EloScoringModel
from src.models.greedy_ranking_model import GreedyRankingModel
from src.models.model_base import ModelBase
from src.scripts.model_types import (
    DenseNetworkSpecification,
    SimpleScoringSpecification,
    EloScoringSpecification,
    GreedyRankingSpecification,
)
from src.scripts.training_spec import TrainingSpecification
from src.utils import data_split


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
    model = _create_starting_model(spec)

    training_data = _load_lmarena_human_preference()
    if spec.data.max_samples is not None:
        downsampled = data_split.downsample(training_data, spec.data.max_samples, spec.data.seed)
        print(f"Downsampled data size: {len(downsampled.entries)}")
    else:
        downsampled = training_data

    model.train(
        downsampled, 
        validation_split=data_split.ValidationSplit(
            val_fraction=spec.data.valiation_split, 
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
        case "simple_scoring":
            return _create_starting_simple_scoring(spec)
        case "elo_scoring":
            return _create_starting_elo_scoring(spec)
        case "greedy_ranking":
            return _create_starting_greedy_ranking(spec)
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


def _load_lmarena_human_preference() -> TrainingData:
    dataset = datasets.load_dataset("lmarena-ai/arena-human-preference-140k")
    training_data = data_loading.load_training_data(dataset["train"].to_pandas())
    print(f"Successfully loaded {len(training_data.entries)} entries")
    return training_data

