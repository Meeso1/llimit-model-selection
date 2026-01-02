from pathlib import Path
from typing import ClassVar
from src.constants import (
    MODELS_JAR_PATH, PREPROCESSED_DATA_JAR_PATH, INFERENCE_OUTPUTS_PATH, 
    MODELS_JAR_DIR_NAME, PREPROCESSED_DATA_JAR_DIR_NAME, INFERENCE_OUTPUTS_JAR_DIR_NAME
)
from src.utils.jar import Jar


class Jars:
    models: ClassVar[Jar] = Jar(MODELS_JAR_PATH)
    preprocessed_data: ClassVar[Jar] = Jar(PREPROCESSED_DATA_JAR_PATH)
    inference_outputs: ClassVar[Jar] = Jar(INFERENCE_OUTPUTS_PATH)

    @staticmethod
    def set_base_path(base_path: Path) -> None:
        Jars.models = Jar(base_path / MODELS_JAR_DIR_NAME)
        Jars.preprocessed_data = Jar(base_path / PREPROCESSED_DATA_JAR_DIR_NAME)
        Jars.inference_outputs = Jar(base_path / INFERENCE_OUTPUTS_JAR_DIR_NAME)
