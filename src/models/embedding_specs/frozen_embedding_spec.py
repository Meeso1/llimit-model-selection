"""Specification for frozen embedding model."""

from typing import Literal

from src.models.embedding_specs.embedding_spec import EmbeddingModelSpecification
from src.models.embedding_models.triplet_frozen_encoder_model import TripletFrozenEncoderModel
from src.models.optimizers.optimizer_spec_union import OptimizerSpec


class FrozenEmbeddingSpec(EmbeddingModelSpecification):
    """
    Specification for frozen sentence transformer embedding model.
    
    Uses a frozen sentence transformer with trainable dense layers on top.
    Fast to train, good baseline performance.
    """
    
    embedding_type: Literal["frozen"] = "frozen"
    encoder_model_name: str = "all-MiniLM-L6-v2"
    hidden_dims: list[int] = [256, 128]
    optimizer: OptimizerSpec
    
    triplet_margin: float = 0.2
    regularization_weight: float = 0.01
    identity_positive_ratio: float = 0.8
    
    def create_model(
        self,
        min_model_comparisons: int,
        preprocessor_seed: int,
        print_every: int | None,
    ) -> TripletFrozenEncoderModel:
        return TripletFrozenEncoderModel(
            encoder_model_name=self.encoder_model_name,
            hidden_dims=self.hidden_dims,
            optimizer_spec=self.optimizer,
            triplet_margin=self.triplet_margin,
            regularization_weight=self.regularization_weight,
            min_model_comparisons=min_model_comparisons,
            identity_positive_ratio=self.identity_positive_ratio,
            preprocessor_seed=preprocessor_seed,
            print_every=print_every,
        )
