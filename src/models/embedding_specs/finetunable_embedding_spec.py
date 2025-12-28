"""Specification for fine-tunable embedding model."""

from typing import Literal

from src.models.embedding_specs.embedding_spec import EmbeddingModelSpecification
from src.models.triplet_finetunable_encoder_model import TripletFinetunableEncoderModel
from src.models.optimizers.optimizer_spec_union import OptimizerSpec


class FinetunableEmbeddingSpec(EmbeddingModelSpecification):
    """
    Specification for fine-tunable transformer embedding model.
    
    Uses a fine-tunable transformer (e.g., BERT) with projection layer.
    More powerful but slower and more resource-intensive than frozen model.
    """
    
    embedding_type: Literal["finetunable"] = "finetunable"
    transformer_model_name: str = "bert-base-uncased"
    projection_dim: int = 128
    max_length: int = 256
    optimizer: OptimizerSpec
    
    triplet_margin: float = 0.2
    regularization_weight: float = 0.01
    identity_positive_ratio: float = 0.8
    
    def create_model(
        self,
        min_model_comparisons: int,
        preprocessor_seed: int,
        print_every: int | None,
    ) -> TripletFinetunableEncoderModel:
        return TripletFinetunableEncoderModel(
            transformer_model_name=self.transformer_model_name,
            projection_dim=self.projection_dim,
            max_length=self.max_length,
            optimizer_spec=self.optimizer,
            triplet_margin=self.triplet_margin,
            regularization_weight=self.regularization_weight,
            min_model_comparisons=min_model_comparisons,
            identity_positive_ratio=self.identity_positive_ratio,
            preprocessor_seed=preprocessor_seed,
            print_every=print_every,
        )
