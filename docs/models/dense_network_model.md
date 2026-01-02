# Dense Network Model

## Overview

The `DenseNetworkModel` is a feedforward neural network designed for prompt routing to LLMs. It learns to score individual models for given prompts, allowing flexible comparison of any number of models at inference time.

## Architecture

### Model Structure

- **Input**: 
  - Fixed-size prompt embedding (default: 384-dimensional from `all-MiniLM-L6-v2`)
  - Prompt features (45-dimensional scalar features extracted from prompt text)
  - Model ID (integer, embedded into learned vector)
- **Model ID Embedding**: Learned embeddings for each model (default: 32-dimensional)
- **Concatenation**: Prompt embedding + Prompt features + Model ID embedding
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

The preprocessor handles a single dataset at a time:

1. **Filtering**: Removes entries with `winner="tie"` or `winner="both_bad"`
2. **Model Encoding**: Creates and fits a `StringEncoder` with all unique model names from the dataset
3. **Feature Extraction**: Extracts 45 scalar features from each prompt (task type, complexity, domain, style, context, output format)
4. **Embedding**: Uses Sentence Transformers to embed user prompts into fixed-size vectors (returned as numpy arrays)
5. **Model ID Assignment**: Converts model names to integer IDs using the encoder
6. **Caching**: Results are cached for reuse (see below)

**Train/Val Split**: The split happens *after* preprocessing on the preprocessed pairs, so both train and val naturally share the same model encoder.

#### Inference Preprocessing

1. **Feature Extraction**: Extracts 45 scalar features from each prompt (task type, complexity, domain, style, context, output format)
2. **Embedding**: Embeds the input prompts (returned as numpy arrays)
3. **Model ID Encoding**: Converts requested model names to IDs using the saved encoder
4. **No Caching**: Inference preprocessing is done on-the-fly

**Note**: During inference, conversation history is not available, so context features are computed with empty history.

#### Caching

Preprocessing is cached per dataset. The cache key is based on:
- Preprocessor version (e.g., "v2")
- Content signature of the dataset (hash of sample entries)

When you preprocess the same dataset (even with different train/val splits), the cached results are reused. This significantly speeds up repeated training runs.

**Cache invalidation**: Changing the dataset content or preprocessor version will trigger re-preprocessing. The version was incremented to v2 when prompt features were added.

### Embedding Models

The default embedding model is `all-MiniLM-L6-v2` from Sentence Transformers:
- **Dimension**: 384
- **Speed**: Fast inference
- **Quality**: Good balance for semantic similarity tasks

Other models can be used by passing `embedding_model_name` to the constructor. Popular alternatives from the RouteLLM paper:
- `all-mpnet-base-v2`: Higher quality, 768 dimensions
- `all-MiniLM-L12-v2`: Medium quality/speed tradeoff, 384 dimensions

### Prompt Features

In addition to semantic embeddings, the model uses 45 scalar features extracted from prompt text. These features capture various aspects of prompts that help the model understand task requirements:

- **Task Type Indicators (10 features)**: Detects code requests, math/reasoning, creative writing, factual queries, instructions, roleplay, and analysis tasks
- **Prompt Complexity (8 features)**: Measures length, vocabulary complexity, sentence structure, nesting depth, and multi-part requests
- **Domain Indicators (12 features)**: Identifies domain-specific content (science, medicine, law, finance, tech, academic, casual, formal, philosophical, historical, personal, business)
- **Linguistic Style (6 features)**: Captures formality, imperative vs interrogative, specificity, politeness, and urgency markers
- **Context Features (4 features)**: Uses conversation history to detect follow-up questions, context length, and dialogue turns
- **Output Format (5 features)**: Identifies expected response formats (list, table, JSON, code, long-form)

These features are extracted using functions in `src/preprocessing/scoring_feature_extraction.py` and concatenated with the prompt embedding before being fed into the neural network. During training, conversation history is available from the dataset. During inference, context features are computed with empty history since conversation context is not provided.

## Usage

### Initialization

```python
from src.models.dense_network_model import DenseNetworkModel
from src.models.optimizers.adamw_spec import AdamWSpec

model = DenseNetworkModel(
    embedding_model_name="all-MiniLM-L6-v2",
    hidden_dims=[256, 128, 64],
    model_id_embedding_dim=32,
    optimizer_spec=AdamWSpec(learning_rate=0.001, lr_decay_gamma=0.95),
    balance_model_samples=True,
)
```

### Training

```python
from src.data_models.data_models import TrainingData
from src.utils.data_split import ValidationSplit

# Training without validation
model.train(training_data, epochs=10, batch_size=32)

# Training with validation split
model.train(
    training_data,
    validation_split=ValidationSplit(val_fraction=0.2, seed=42),
    epochs=10,
    batch_size=32,
)
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
- **PreprocessedPromptPair**: Cached embedding with prompt features, model IDs and winner label
- **PreprocessedTrainingData**: Collection of preprocessed pairs with string encoder, embedding dimension, and prompt features dimension
- **PreprocessedInferenceInput**: Embedded prompts, prompt features, and model IDs for inference

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
- `optimizer_spec`: Optimizer specification (default: AdamW with LR 0.001)
  - See `docs/optimizers.md` for details on configuring optimizers and LR decay
- `balance_model_samples`: Whether to balance model representation (default: True)
  - See `docs/sample_balancing.md` for details

### Training Parameters

- `epochs`: Number of training epochs (default: 10)
- `batch_size`: Batch size for training (default: 32)
- `margin`: Margin for ranking loss (fixed at 0.1)

### Dropout

Fixed at 0.2 in hidden layers to prevent overfitting.

## Implementation Details

### Device Management

The model automatically detects and uses CUDA if available, falling back to CPU otherwise.

### Initialization and Training Flow

Training follows a straightforward flow:

1. **`_initialize_and_preprocess(data)`**: Called once at the start of training
   - Takes the full training dataset
   - Initializes Weights & Biases
   - Preprocesses the entire dataset (embeddings and model encoding)
   - Creates and stores the model encoder
   - Initializes neural network with correct dimensions
   - Returns preprocessed data

2. **Train/Val Split**: After preprocessing, if validation is requested
   - Uses `split_preprocessed_data()` to split the preprocessed pairs
   - Both splits share the same model encoder (ensures consistent model IDs)
   - Splitting happens on preprocessed data, not raw data

3. **`_prepare_dataloader(preprocessed_data, ...)`**: Called for train and val
   - Takes already-preprocessed data
   - Converts preprocessed pairs to PyTorch tensors
   - Creates TensorDataset
   - Optionally applies balanced sampling (training only)
   - Returns configured DataLoader

This approach ensures the model encoder is fitted on the full dataset while keeping preprocessing and splitting as separate, simple steps.

### Network Architecture (Inner Class)

The actual PyTorch module is implemented as `_DenseNetwork`, an inner class at the bottom of the model class. Key components:

1. **Model Embedding Layer**: `nn.Embedding(num_models, model_id_embedding_dim)` - learns representations for each model
2. **Concatenation**: Combines prompt embedding, prompt features, and model ID embedding
3. **Dense Layers**: Standard feedforward layers with ReLU and dropout
4. **Output**: Single score per (prompt, model) pair

### Preprocessing Cache Location

Preprocessed data is saved to `preprocessed_data/` directory in the project root (configured via `PREPROCESSED_DATA_JAR_PATH` in `src/constants.py`).

### Training Data Expansion

During training, each comparison pair (prompt, model_a, model_b, winner) is converted into a ranking example:
- score(prompt, model_a) and score(prompt, model_b) are computed
- Margin ranking loss ensures the winner gets a higher score

This is more data-efficient than creating separate positive/negative examples.

## Optimizer and Training Features

### Configurable Optimizers

The model supports multiple optimizer types (Adam, AdamW, Muon) with configurable parameters including learning rate decay. See `docs/optimizers.md` for:
- Available optimizer types and their parameters
- How to configure exponential LR decay
- Serialization/deserialization for model saving

### Sample Balancing

The model can automatically balance model representation during training to handle dataset imbalance. See `docs/sample_balancing.md` for:
- How weighted sampling works
- When to enable/disable balancing
- Implementation details and limitations

## Limitations

- Can only score models that were present in the training data (string encoder limitation)
- Ties and "both_bad" entries are filtered out (not used for training)
- Fixed embedding dimension per model instance (determined by embedding model)
- Fixed prompt features dimension (45 features)
- Single score output (doesn't model uncertainty or provide probability distributions)
- Model ID embeddings require sufficient training data per model to learn meaningful representations
- Preprocessing cache is version-based only, not data-specific (manual cache management needed if dataset changes)
- Sample balancing uses inverse frequency weighting and doesn't handle extreme cases (single occurrence)
- Conversation history is not available during inference, so context features use empty history
