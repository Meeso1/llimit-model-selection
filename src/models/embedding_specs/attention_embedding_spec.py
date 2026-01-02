"""Specification for attention-based embedding model."""

from typing import Literal

from src.models.embedding_specs.embedding_spec import EmbeddingModelSpecification
from src.models.embedding_models.attention_embedding_model import AttentionEmbeddingModel
from src.models.optimizers.optimizer_spec_union import OptimizerSpec


class AttentionEmbeddingSpec(EmbeddingModelSpecification):
    """
    Specification for attention-based embedding model.
    
    Uses attention pooling over (prompt, response) pairs to create model embeddings.
    Learns with supervised contrastive loss.
    """
    
    embedding_type: Literal["attention"] = "attention"
    encoder_model_name: str = "all-MiniLM-L6-v2"
    h_emb: int = 256
    h_scalar: int = 64
    h_pair: int = 256
    d_out: int = 128
    pair_mlp_layers: int = 2
    num_attention_heads: int = 4
    dropout: float = 0.1
    temperature: float = 0.07
    pairs_per_model: int = 32
    models_per_batch: int = 16
    embeddings_per_model: int = 4
    optimizer: OptimizerSpec
    
    def create_model(
        self,
        min_model_comparisons: int,
        preprocessor_seed: int,
        print_every: int | None,
    ) -> AttentionEmbeddingModel:
        return AttentionEmbeddingModel(
            embedding_model_name=self.encoder_model_name,
            h_emb=self.h_emb,
            h_scalar=self.h_scalar,
            h_pair=self.h_pair,
            d_out=self.d_out,
            pair_mlp_layers=self.pair_mlp_layers,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            temperature=self.temperature,
            optimizer_spec=self.optimizer,
            min_model_comparisons=min_model_comparisons,
            preprocessor_seed=preprocessor_seed,
            pairs_per_model=self.pairs_per_model,
            models_per_batch=self.models_per_batch,
            embeddings_per_model=self.embeddings_per_model,
            print_every=print_every,
        )

