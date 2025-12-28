"""Union type for embedding model specifications."""

from typing import Annotated, Union
from pydantic import Field

from src.models.embedding_specs.frozen_embedding_spec import FrozenEmbeddingSpec
from src.models.embedding_specs.finetunable_embedding_spec import FinetunableEmbeddingSpec
from src.models.embedding_specs.attention_embedding_spec import AttentionEmbeddingSpec

EmbeddingSpec = Annotated[
    Union[FrozenEmbeddingSpec, FinetunableEmbeddingSpec, AttentionEmbeddingSpec],
    Field(discriminator="embedding_type")
]
