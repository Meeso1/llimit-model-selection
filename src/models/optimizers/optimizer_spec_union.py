from typing import Union, Annotated
from pydantic import Field

from src.models.optimizers.adam_spec import AdamSpec
from src.models.optimizers.adamw_spec import AdamWSpec
from src.models.optimizers.muon_spec import MuonSpec

OptimizerSpec = Annotated[Union[AdamSpec, AdamWSpec, MuonSpec], Field(discriminator="optimizer_type")]

