# Dense Network Model

## Overview

The `DenseNetworkModel` is a feedforward neural network designed for prompt routing to LLMs. It learns to score individual models for given prompts, allowing flexible comparison of any number of models at inference time.

## Architecture

### Model Structure

- **Input**: 
  - Fixed-size prompt embedding (default: 384-dimensional from `all-MiniLM-L6-v2`)
  - Model ID (integer, embedded into learned vector)
- **Model ID Embedding**: Learned embeddings for each model (default: 32-dimensional)
- **Concatenation**: Prompt embedding + Model ID embedding
- **Hidden Layers**: Configurable dense layers with ReLU activation and dropout (default: [256, 128, 64])
- **Output**: Single score value constrained to [-1, 1] using tanh activation

### Training Objective

The model is trained using Margin Ranking Loss to ensure winning models get higher scores:
- For each comparison in training data, we create a pair: (prompt, model_a) and (prompt, model_b)
- The loss encourages score(prompt, winner) > score(prompt, loser) + margin
- Margin is set to 0.1 by default

This approach allows the model to learn absolute scores for (prompt, model) pairs rather than just pairwise comparisons.

### Training Metrics

During training, both loss and accuracy are computed and logged:

- **Loss**: Margin ranking loss (lower is better)
- **Accuracy**: Percentage of pairs where the model's prediction matches human evaluation (higher is better)
  - See `docs/accuracy_metric.md` for detailed explanation

Both train and validation metrics are tracked in the training history and can be logged to Weights & Biases.

## String Encoder

### StringEncoder

The `StringEncoder` class (in `src/utils/string_encoder.py`) is a general-purpose utility for mapping strings to integer IDs. In this model, it's used to encode model names:

- Strings are assigned IDs in the order they are first encountered (sorted alphabetically for determinism)
- The encoder is fitted during training data preprocessing
- It's saved with the model for use during inference
- Allows the neural network to work with integer IDs rather than string names
- Returns `None` for unknown strings (doesn't raise errors)

Example:
```python
encoder = StringEncoder()
encoder.fit(["gpt-4", "gpt-3.5-turbo", "claude-2"])
encoder.encode(["gpt-4"])  # Returns [1]
encoder.encode(["unknown"])  # Returns [None]
encoder.decode([0])  # Returns ["claude-2"]
encoder.encode_known(["gpt-4", "unknown"])  # Returns ([1], ["gpt-4"])
```

For more details on `StringEncoder`, see the preprocessing documentation.

## Preprocessing

### PromptEmbeddingPreprocessor

The `PromptEmbeddingPreprocessor` handles data preparation for both training and inference:

#### Training Preprocessing

1. **Filtering**: Removes entries with `winner="tie"` or `winner="both_bad"` from training data
2. **Model Encoding**: Creates and fits a `StringEncoder` with all unique model names from training data
3. **Embedding**: Uses Sentence Transformers to embed user prompts into fixed-size vectors (returned as numpy arrays)
4. **Model ID Assignment**: Converts model names to integer IDs using the encoder
5. **Caching**: Preprocessed data (including the encoder) is cached using the Jar system

#### Inference Preprocessing

1. **Embedding**: Embeds the input prompts (returned as numpy arrays)
2. **Model ID Encoding**: Converts requested model names to IDs using the saved encoder
3. **No Caching**: Inference preprocessing is done on-the-fly

#### Cache Structure

Cache keys follow the format: `prompt_embedding/{version}`

- `prompt_embedding`: Identifies the preprocessor type
- `version`: Preprocessor version (currently fixed at "v1")

The cache is version-based only, not data-specific. This means:
- When the version changes, preprocessing is re-run
- The same cache is reused for all training runs with the same version
- To force re-preprocessing, manually clear the cache or use a different version

### Embedding Models

The default embedding model is `all-MiniLM-L6-v2` from Sentence Transformers:
- **Dimension**: 384
- **Speed**: Fast inference
- **Quality**: Good balance for semantic similarity tasks

Other models can be used by passing `embedding_model_name` to the constructor. Popular alternatives from the RouteLLM paper:
- `all-mpnet-base-v2`: Higher quality, 768 dimensions
- `all-MiniLM-L12-v2`: Medium quality/speed tradeoff, 384 dimensions

## Usage

### Initialization

```python
from src.models.dense_network_model import DenseNetworkModel

model = DenseNetworkModel(
    embedding_model_name="all-MiniLM-L6-v2",
    hidden_dims=[256, 128, 64],
    model_id_embedding_dim=32,
    learning_rate=0.001,
)
```

### Training

```python
from src.data_models.data_models import TrainingData

# Assuming you have loaded training_data
model.train(training_data, epochs=10, batch_size=32)
```

### Prediction

The model can score any number of models for given prompts:

```python
from src.data_models.data_models import InputData

# Score multiple models for multiple prompts
input_data = InputData(
    prompts=[
        "How do I sort a list in Python?",
        "Explain quantum computing",
    ],
    model_names=["gpt-3.5-turbo", "gpt-4", "claude-2"],
)

output = model.predict(input_data)
# output.scores is a dict: {"gpt-3.5-turbo": array([...]), "gpt-4": array([...]), ...}
# Each array has shape [n_prompts]
# Higher score means model is more suitable for that prompt

# Example: Route first prompt to best model
scores_for_prompt_0 = {name: scores[0] for name, scores in output.scores.items()}
best_model = max(scores_for_prompt_0, key=scores_for_prompt_0.get)
```

### Saving and Loading

```python
# Save (includes string encoder)
model.save("my_dense_model")

# Load (restores string encoder)
loaded_model = DenseNetworkModel.load("my_dense_model")

# The loaded model can score the same models it was trained on
input_data = InputData(
    prompts=["New prompt"],
    model_names=["gpt-4", "gpt-3.5-turbo"],  # Must be models from training data
)
output = loaded_model.predict(input_data)
```

## Data Models

### Input/Output Types

Model-invariant types (in `src/data_models/data_models.py`):
- **InputData**: Model-agnostic input containing prompts and model names
- **OutputData**: Abstract base class with `scores` property that all models must implement

Model-specific types (in `src/data_models/dense_network_types.py`):
- **PromptRoutingOutput**: Inherits from `OutputData`, contains dictionary mapping model names to score arrays
- **PreprocessedPromptPair**: Cached embedding with model IDs and winner label
- **PreprocessedTrainingData**: Collection of preprocessed pairs with string encoder
- **PreprocessedInferenceInput**: Embedded prompts and model IDs for inference

### Score Interpretation

Scores are in the range [-1, 1]:
- **Higher score**: Model is more suitable for the prompt
- **Lower score**: Model is less suitable for the prompt
- **Magnitude**: Confidence (closer to ±1 = higher confidence)

Unlike pairwise comparison models, this model produces absolute scores for each (prompt, model) combination. This allows:
- Scoring any number of models simultaneously
- Comparing models that weren't directly compared in training
- Using the same model across different routing scenarios

## Configuration

### Hyperparameters

- `embedding_model_name`: Sentence transformer model name (default: "all-MiniLM-L6-v2")
- `hidden_dims`: List of hidden layer sizes (default: [256, 128, 64])
- `model_id_embedding_dim`: Dimension of learned model ID embeddings (default: 32)
- `learning_rate`: Adam optimizer learning rate (default: 0.001)

### Training Parameters

- `epochs`: Number of training epochs (default: 10)
- `batch_size`: Batch size for training (default: 32)
- `margin`: Margin for ranking loss (fixed at 0.1)

### Dropout

Fixed at 0.2 in hidden layers to prevent overfitting.

## Implementation Details

### Device Management

The model automatically detects and uses CUDA if available, falling back to CPU otherwise.

### Network Architecture (Inner Class)

The actual PyTorch module is implemented as `_DenseNetwork`, an inner class at the bottom of the model class. Key components:

1. **Model Embedding Layer**: `nn.Embedding(num_models, model_id_embedding_dim)` - learns representations for each model
2. **Concatenation**: Combines prompt embedding with model ID embedding
3. **Dense Layers**: Standard feedforward layers with ReLU and dropout
4. **Output**: Single score per (prompt, model) pair

### Preprocessing Cache Location

Preprocessed data is saved to `preprocessed_data/` directory in the project root (configured via `PREPROCESSED_DATA_JAR_PATH` in `src/constants.py`).

### Training Data Expansion

During training, each comparison pair (prompt, model_a, model_b, winner) is converted into a ranking example:
- score(prompt, model_a) and score(prompt, model_b) are computed
- Margin ranking loss ensures the winner gets a higher score

This is more data-efficient than creating separate positive/negative examples.

## Limitations

- Can only score models that were present in the training data (string encoder limitation)
- Ties and "both_bad" entries are filtered out (not used for training)
- Fixed embedding dimension per model instance (determined by embedding model)
- Single score output (doesn't model uncertainty or provide probability distributions)
- Model ID embeddings require sufficient training data per model to learn meaningful representations
- Preprocessing cache is version-based only, not data-specific (manual cache management needed if dataset changes)

