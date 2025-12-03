from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_JAR_PATH = PROJECT_ROOT / "saved_models"
PREPROCESSED_DATA_JAR_PATH = PROJECT_ROOT / "preprocessed_data"
INFERENCE_OUTPUTS_PATH = PROJECT_ROOT / "inferrence_outputs"