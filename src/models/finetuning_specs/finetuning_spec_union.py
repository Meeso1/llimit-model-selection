"""Union type for fine-tuning specifications."""

from typing import Annotated, Union
from pydantic import Field

from src.models.finetuning_specs.lora_spec import LoraSpec
from src.models.finetuning_specs.qlora_spec import QLoraSpec
from src.models.finetuning_specs.last_layers_spec import LastLayersSpec
from src.models.finetuning_specs.bitfit_spec import BitFitSpec
from src.models.finetuning_specs.full_finetuning_spec import FullFineTuningSpec

FineTuningSpec = Annotated[
    Union[LoraSpec, QLoraSpec, LastLayersSpec, BitFitSpec, FullFineTuningSpec],
    Field(discriminator="method")
]

