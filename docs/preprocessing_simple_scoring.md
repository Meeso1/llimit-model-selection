# Simple Scoring Preprocessor

## Overview

The `SimpleScoringPreprocessor` prepares training data for the `SimpleScoringModel` by extracting model comparisons and encoding model names to integer IDs.

## Input/Output

### Input
- `TrainingData`: Contains `EvaluationEntry` objects with model comparisons

### Output
- `PreprocessedTrainingData`: Contains:
  - `comparisons`: List of `PreprocessedComparison` objects
  - `model_encoder`: `StringEncoder` mapping model names to IDs

## Data Structures

### PreprocessedComparison
```python
@dataclass
class PreprocessedComparison:
    model_id_a: int  # Encoded model A ID
    model_id_b: int  # Encoded model B ID
    comparison_type: Literal["model_a_wins", "model_b_wins", "tie", "both_bad"]
```

### PreprocessedTrainingData
```python
@dataclass
class PreprocessedTrainingData:
    comparisons: list[PreprocessedComparison]
    model_encoder: StringEncoder
```

## Comparison Type Mapping

The preprocessor maps `EvaluationEntry.winner` values to comparison types:

| Winner Value | Comparison Type | Meaning |
|--------------|----------------|---------|
| `"model_a"` | `"model_a_wins"` | Model A performed better |
| `"model_b"` | `"model_b_wins"` | Model B performed better |
| `"tie"` | `"tie"` | Both models performed equally well |
| `"both_bad"` | `"both_bad"` | Both models performed poorly |

Any other winner value is skipped during preprocessing.

## Usage

```python
from src.preprocessing.simple_scoring_preprocessor import SimpleScoringPreprocessor
from src.data_models.data_models import TrainingData

preprocessor = SimpleScoringPreprocessor()
preprocessed_data = preprocessor.preprocess(training_data)

# Access model encoder
model_encoder = preprocessed_data.model_encoder
print(f"Number of unique models: {model_encoder.size}")

# Access comparisons
for comparison in preprocessed_data.comparisons[:5]:
    model_a_name = model_encoder.decode(comparison.model_id_a)
    model_b_name = model_encoder.decode(comparison.model_id_b)
    print(f"{model_a_name} vs {model_b_name}: {comparison.comparison_type}")
```

## Model Encoder

The preprocessor creates a `StringEncoder` that:
- Maps model names to unique integer IDs
- Assigns IDs in alphabetical order (deterministic)
- Can encode/decode model names
- Can check if a model name exists

## Implementation Details

1. **Extract Unique Models**: Iterates through all entries to collect unique model names
2. **Create Encoder**: Initializes `StringEncoder` and fits it with sorted model names
3. **Process Comparisons**: For each entry:
   - Encode model_a and model_b to IDs
   - Map winner to comparison type
   - Create `PreprocessedComparison` object
   - Skip entries with unknown winner types
4. **Return**: `PreprocessedTrainingData` with comparisons and encoder

## Design Rationale

The preprocessor is intentionally simple because:
- No embeddings needed (model doesn't look at prompts)
- Only needs to encode model names to IDs
- Handles all comparison types uniformly
- Creates reusable model encoder for inference

This design keeps the preprocessing fast and the model architecture minimal.

