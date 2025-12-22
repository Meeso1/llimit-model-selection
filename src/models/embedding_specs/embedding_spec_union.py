"""Union type for embedding model specifications."""

from typing import Annotated, Union
from pydantic import Field

from src.models.embedding_specs.frozen_embedding_spec import FrozenEmbeddingSpec
from src.models.embedding_specs.finetunable_embedding_spec import FinetunableEmbeddingSpec

EmbeddingSpec = Annotated[
    Union[FrozenEmbeddingSpec, FinetunableEmbeddingSpec],
    Field(discriminator="embedding_type")
]
