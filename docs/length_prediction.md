# Response Length Prediction

This document describes the response length prediction system for LLM model selection.

## Overview

The length prediction system predicts the response length (in tokens) that different LLMs will generate for a given prompt. This is useful for:
- Estimating API costs before making requests
- Selecting models based on desired response verbosity
- Understanding model behavior patterns
- Optimizing for latency-constrained applications

## Data Models

### Input and Output

Length prediction models use the **same input/output format as scoring models**:

- **`InputData`**: Contains prompts and model names for inference
  - `prompts: list[str]` - List of prompts to predict response lengths for
  - `model_names: list[str]` - List of model names to predict for

- **`LengthPredictionOutputData`**: Contains predicted response lengths
  - `predictions: dict[str, np.ndarray]` - Maps model names to predicted lengths `[n_prompts]`

### Training Data

Length prediction models use **`TrainingData`** - the same data format as scoring models. Response lengths are automatically extracted from model responses in the evaluation entries.

## Models

### Base Class: `LengthPredictionModelBase`

All length prediction models inherit from this abstract base class, which provides:
- Training interface: `train(data: TrainingData, validation_split, epochs, batch_size)`
- Prediction interface: `predict(X: InputData, batch_size) -> LengthPredictionOutputData`
- Model persistence: `save(name)` and `load(name)`
- Weights & Biases integration
- Training history tracking

**Key difference from scoring models**: Uses regression (MSE loss) instead of ranking loss.

### Dense Network Model: `DnEmbeddingLengthPredictionModel`

A neural network model that predicts response lengths using:

**Input Features:**
- **Prompt embeddings**: Dense embeddings from sentence transformers (e.g., all-MiniLM-L6-v2)
- **Prompt features**: 45 handcrafted features including:
  - Task type indicators (code, math, creative, etc.)
  - Complexity measures (length, vocabulary, structure)
  - Domain indicators (science, law, tech, etc.)
  - Style features (formality, specificity, politeness)
  - Context features (conversation history)
  - Output format expectations
- **Model embeddings**: Learned embeddings for each LLM (reused from scoring models)
- **Model ID**: Integer index for each LLM, looked up through a learnable `nn.Embedding` layer (dim configurable via `model_id_embedding_dim`, default 8). An extra embedding at index `n_models` serves as the fallback for models not seen during training.

Model embeddings and model IDs are both kept as inputs because they serve complementary roles: model embeddings encode inter-model behavioral relationships learned from pairwise scoring data, while model IDs let the length prediction network learn per-model length patterns independently. If model embeddings turn out to be redundant for a given dataset, the network can learn to down-weight them without losing information.

**Architecture:**
- Each of the three pre-computed inputs is independently projected to `input_proj_dim` (default 64) via `Linear + LeakyReLU(0.1)`, then concatenated with the model ID embedding. This normalises the different raw dimensionalities (e.g., 384-dim sentence-transformer space, varying model embedding dims) into a homogeneous space before mixing.
- The model ID embedding (`nn.Embedding`, dim `model_id_embedding_dim`, default 64) is **not** projected — it is a fully-learned representation whose dimension is directly controlled, so adding a projection would be redundant. Its default is set equal to `input_proj_dim` so each modality starts with equal representation in the concatenated input.
- Concatenated input dimension: `input_proj_dim * 3 + model_id_embedding_dim`.
- Passes through configurable residual blocks (default: [256, 128, 64]):
  - Each block has a **main path** (Linear → LeakyReLU → Dropout) and a **projection shortcut** (bias-free Linear), summed together
  - Shortcut always projects since dimensions change between blocks, enabling direct gradient flow
- Final `Linear(last_hidden_dim, 1)` output layer (no activation — unbounded regression)
- All linear layers are initialised with Kaiming Normal (`a=0.1, nonlinearity='leaky_relu'`).
- Predictions are automatically descaled to original token count range

**Training:**
- Loss function: Mean Squared Error (MSE) on standardized log-length **residuals**
- Optimizer: AdamW (configurable)
- Supports validation splits
- Uses `_scaler` (fits on raw lengths from preprocessing) for both input and output transforms
- **Per-model average lengths** (`_model_avg_lengths`) are computed from the training split at the start of training. The network learns to predict `scaler.transform(log(raw_length)) - model_avg`, i.e. the deviation from each model's baseline verbosity. During inference the per-model average is added back before inverse-scaling.
- Tracks training/validation loss and custom metrics (metrics are always reported in original token-count space)
- Integrates with Weights & Biases
- **Automatically trains embedding model if not initialized**

**Metrics:**
The model tracks several custom metrics to evaluate performance:
- **avg_relative_error**: `1 - abs(1 - predicted/actual)` - "accuracy" metric (higher is better, max 1.0)
- **avg_relative_ratio**: `mean(predicted/actual)` - shows if model tends to over/underpredict (should be ~1.0)
- **stddev_ratio**: `stddev(predictions)/stddev(actuals)` - variance matching (should be ~1.0)
- **mae**: Mean absolute error in original token space

**Per-epoch training diagnostics** (via `EpochDiagnosticsAccumulator`):
- **Gradient norms** (`trunk_grad_norm`, `prompt_emb_proj_grad_norm`, `prompt_feat_proj_grad_norm`, `model_emb_proj_grad_norm`, `model_id_embedding_grad_norm`): L2 gradient norms per parameter group, averaged over batches. Useful for detecting vanishing/exploding gradients or imbalanced update magnitudes across modules.
- **Input projection norms** (`prompt_emb_proj_norm`, `prompt_feat_proj_norm`, `model_emb_proj_norm`): Mean L2 norm of the projected representations for each input modality. Tracks the scale of activations entering the trunk.
- **Input projection variance** (`prompt_emb_proj_variance`, `prompt_feat_proj_variance`, `model_emb_proj_variance`): Mean per-dimension variance of projected representations. Low values signal representation collapse.
- **Gradient × input attribution** (`grad_attr_<FeatureName>` for each of the 45 prompt features, plus `grad_attr_prompt_embedding` and `grad_attr_model_embedding`): Mean `|grad| × |input|` per input, averaged over the batch. Measures how much each input contributes to the loss gradient — a rough feature-importance signal during training.

**Key Parameters:**
```python
model = DnEmbeddingLengthPredictionModel(
    hidden_dims=[256, 128, 64],  # Network architecture
    dropout=0.2,
    input_proj_dim=64,  # Each of the 3 pre-computed inputs is projected to this dim before concatenation
    optimizer_spec=AdamWSpec(learning_rate=0.001),
    embedding_model_name="all-MiniLM-L6-v2",
    load_embedding_model_from="dn_embedding/my_scoring_model",  # Load pre-trained embeddings (format: model_type/model_name)
    min_model_comparisons=20,  # Filter rare models
    embedding_model_epochs=10,
    model_id_embedding_dim=64,  # Dimension of learned model ID embeddings; matches input_proj_dim by default
    print_every=1,  # Print progress every N epochs
    seed=42,
)
```

### Gradient Boosting Model: `GbLengthPredictionModel`

A gradient boosting (XGBoost) model that predicts response lengths using tree-based ensemble methods.

**Input Features:**
Supports configurable input features (via `input_features` parameter):
- **`prompt_features`**: 45 handcrafted features (same as dense network model)
- **`model_embedding`**: Learned embeddings for each LLM
- **`prompt_embedding`**: Dense embeddings from sentence transformers
- **`model_id`**: One-hot encoded model ID (vector of size `n_models`). Unseen models at inference receive an all-zeros vector.

Model embeddings and model IDs are both kept as inputs because they serve complementary roles: model embeddings encode inter-model behavioral relationships learned from pairwise scoring data, while model IDs let XGBoost learn per-model length patterns via direct binary splits. Users can experiment with subsets of features via the `input_features` parameter.

**Architecture:**
- Uses XGBoost with regression objective (`reg:squarederror`)
- Trains multiple decision trees sequentially (boosting rounds = epochs)
- Each tree learns from the residuals of previous trees
- Features are concatenated based on `input_features` configuration
- Predictions are automatically descaled to original token count range

**Training:**
- Loss function: Mean Squared Error (RMSE reported) on standardized log-length **residuals**
- Configurable XGBoost hyperparameters (depth, learning rate, regularization)
- Supports validation splits
- Uses `_scaler` (fits on raw lengths from preprocessing) for both input and output transforms
- **Per-model average lengths** (`_model_avg_lengths`) are computed from the training split at the start of training. XGBoost trees learn to predict `scaler.transform(log(raw_length)) - model_avg`; predictions are converted back to raw token counts via `model_avg + output → scaler.inverse_transform → exp`.
- Metrics are always reported in original token-count space
- **Automatically trains embedding model if not initialized**
- **Implements best model tracking**: Reverts to epoch with highest validation accuracy after training

**Best Model Tracking:**
- After each epoch, the model state is saved if it achieves better validation accuracy
- At the end of training, the model automatically reverts to the best epoch
- Uses validation accuracy if available, otherwise train accuracy

**Metrics:**
The model tracks several custom metrics:
- **relative_accuracy**: `1 - abs(1 - predicted/actual)` - "accuracy" metric (higher is better, max 1.0)
- **rmse**: Root Mean Squared Error in scaled space
- **mae**: Mean absolute error in original token space

**Per-epoch training diagnostics:**
- **`train_prediction_std`**: Standard deviation of predictions across all training samples at the current round. Tracks how the model's output distribution spreads as more trees are added.
- **`tree_contribution_mean`**: Mean absolute delta between current and previous round's predictions (unavailable at round 1). Measures how much the most recently added tree moved predictions — a convergence signal; should decrease toward zero in later rounds.
- **Feature block importance** (`importance_prompt_features`, `importance_model_embedding`, `importance_prompt_embedding`, `importance_model_id`): Fraction of total XGBoost `total_gain` attributable to each input feature block. Values sum to 1.0 across blocks.
- **Per-feature importances** (`importance_feature/<FeatureName>` for each of the 45 named prompt features): Fraction of total gain for each individual named prompt feature. These are stored separately for analysis but the plot aggregates them into the block-level `importance_prompt_features` bar.

**Key Parameters:**
```python
model = GbLengthPredictionModel(
    max_depth=6,  # Maximum tree depth
    learning_rate=0.1,  # Boosting learning rate
    colsample_bytree=1.0,  # Fraction of features per tree (column sampling)
    colsample_bylevel=1.0,  # Fraction of features per level (feature-level sampling)
    reg_alpha=0.0,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    input_features=["prompt_features", "model_embedding", "prompt_embedding", "model_id"],  # Configurable inputs
    embedding_spec=FrozenEmbeddingSpec(...),  # Or load_embedding_model_from
    embedding_model_name="all-MiniLM-L6-v2",
    min_model_comparisons=20,
    embedding_model_epochs=10,
    print_every=1,
    seed=42,
)
```

**When to use:**
- When you want a non-neural, tree-based approach
- When you have limited data (gradient boosting can work well with smaller datasets)
- When you want interpretable feature importance
- When you want flexibility in choosing input features (can use subset of available features)

## Preprocessing

### `LengthPredictionPreprocessor`

Handles data preprocessing for length prediction:

**Process:**
1. Wraps `PromptEmbeddingPreprocessor` internally for prompt embeddings and features
2. Extracts response lengths from `model_a_response` and `model_b_response` fields
3. Approximates token counts using heuristic: `word_count * 1.3`
4. Fits `SimpleScaler` on all lengths (standardizes to mean=0, stddev=1)
5. Stores scaler state for inverse transform during inference
6. Caches preprocessed data for efficiency

**Two-stage preprocessing:**
- **First stage**: Returns `PreprocessedLengthPredictionTrainingData` (without model embeddings)
  - Contains scaled lengths and scaler state
  - Ready for model embedding addition
- **Second stage**: Call `add_model_embeddings(model_embeddings, model_embedding_dim)` 
  - Creates `PreprocessedLengthPredictionTrainingDataWithEmbeddings`

This design avoids placeholder values and ensures model embeddings are only added after the embedding model is loaded.

**Usage:**
```python
preprocessor = LengthPredictionPreprocessor(
    embedding_model_name="all-MiniLM-L6-v2",
    min_model_comparisons=20,
)

# Training preprocessing
preprocessed_without_embeddings = preprocessor.preprocess(training_data)
preprocessed_data = preprocessed_without_embeddings.add_model_embeddings(model_embeddings)

# Inference preprocessing
inference_input = preprocessor.preprocess_for_inference(
    prompts=["Write a story about..."],
    model_names=["gpt-4", "claude-3"],
    model_encoder=model_encoder,
)
```

## Model Embeddings

Length prediction models **reuse model embeddings from scoring models**. The embeddings capture model behavior characteristics learned from pairwise comparisons.

**Two options:**

1. **Load pre-trained embeddings**:
```python
length_model = DnEmbeddingLengthPredictionModel(
    load_embedding_model_from="dn_embedding/my_dn_embedding_model",  # Format: model_type/model_name
    # ... other parameters
)
```

2. **Train embeddings from scratch** (if not initialized):
```python
length_model = DnEmbeddingLengthPredictionModel(
    embedding_spec=FrozenEmbeddingSpec(...),  # Will train if needed
    # ... other parameters
)
```

The model automatically:
- Loads embedding model from source if specified
- **Trains embedding model on the data if not initialized** (like scoring models do)
- Extracts model embeddings when calling `add_model_embeddings(model_embeddings, embedding_dim)` during preprocessing

## Usage

### CLI Training

Length prediction models use the same training command as scoring models. Create a JSON specification file:

```json
{
  "model": {
    "name": "my_length_predictor",
    "spec": {
      "model_type": "dn_embedding_length_prediction",
      "hidden_dims": [256, 128, 64],
      "input_proj_dim": 64,
      "optimizer": {
        "optimizer_type": "adamw",
        "learning_rate": 0.001
      },
      "load_embedding_model_from": "dn_embedding/my_scoring_model",
      "embedding_model_name": "all-MiniLM-L6-v2",
      "min_model_comparisons": 20,
      "embedding_model_epochs": 10,
      "seed": 42
    }
  },
  "data": {
    "dataset": "both",
    "validation_split": 0.2,
    "seed": 42
  },
  "epochs": 50,
  "batch_size": 32,
  "log": {
    "print_every": 1
  }
}
```

Train the model:
```bash
uv run python -m src.scripts.cli train --spec-file length_prediction_spec.json
```

### CLI Inference

Use the same inference command as scoring models:

```bash
uv run python -m src.scripts.cli infer \
  --model "dn_embedding_length_prediction/my_length_predictor" \
  --models-to-score "gpt-4" "claude-3" "llama-2" \
  --prompts "Write a short poem" "Explain quantum computing"
```

The output JSON will include `model_kind` field to distinguish between scoring and length prediction results:

```json
{
  "model_kind": "length_prediction",
  "result_type": "predicted_lengths",
  "results": {
    "gpt-4": [150.5, 320.8],
    "claude-3": [180.2, 280.4],
    "llama-2": [120.3, 250.6]
  }
}
```

### API

**Endpoint**: `POST /infer`

The API uses a unified endpoint that can run both scoring and length prediction. To use length prediction, specify the `length_prediction_model` parameter.

**Request (length prediction only):**
```json
{
  "length_prediction_model": "dn_embedding_length_prediction/my_model",
  "model_names": ["gpt-4", "claude-3", "llama-2"],
  "prompts": ["Write a short poem", "Explain quantum computing"],
  "batch_size": 128
}
```

**Response:**
```json
{
  "scores": null,
  "predicted_lengths": {
    "gpt-4": [150.5, 320.8],
    "claude-3": [180.2, 280.4],
    "llama-2": [120.3, 250.6]
  }
}
```

**Request (both scoring and length prediction):**
```json
{
  "scoring_model": "dn_embedding/my_scoring_model",
  "length_prediction_model": "dn_embedding_length_prediction/my_model",
  "model_names": ["gpt-4", "claude-3", "llama-2"],
  "prompts": ["Write a short poem", "Explain quantum computing"],
  "batch_size": 128
}
```

**Response:**
```json
{
  "scores": {
    "gpt-4": [0.8, 0.7],
    "claude-3": [0.7, 0.75],
    "llama-2": [0.6, 0.65]
  },
  "predicted_lengths": {
    "gpt-4": [150.5, 320.8],
    "claude-3": [180.2, 280.4],
    "llama-2": [120.3, 250.6]
  }
}
```

### Python API Example

```python
from src.data_models.data_models import TrainingData, InputData
from src.models.length_prediction.dn_embedding_length_prediction_model import DnEmbeddingLengthPredictionModel
from src.models.embedding_specs.frozen_embedding_spec import FrozenEmbeddingSpec
from src.models.optimizers.adamw_spec import AdamWSpec
from src.utils.data_split import ValidationSplit

# Prepare training data (same format as scoring models)
training_data = TrainingData(entries=[...])  # Your evaluation entries

# Create and train model
model = DnEmbeddingLengthPredictionModel(
    load_embedding_model_from="dn_embedding/my_scoring_model",  # Format: model_type/model_name
    hidden_dims=[256, 128, 64],
    print_every=1,
)

model.train(
    data=training_data,
    validation_split=ValidationSplit(val_fraction=0.2, seed=42),
    epochs=50,
    batch_size=32,
)

# Save model
model.save("my_length_predictor")

# Make predictions (same format as scoring models)
input_data = InputData(
    prompts=["Write a short poem about AI"],
    model_names=["gpt-4", "claude-3", "llama-2"],
)

predictions = model.predict(input_data)
for model_name, lengths in predictions.predicted_lengths.items():
    print(f"{model_name}: {lengths[0]:.0f} tokens")
```

## Implementation Notes

### Differences from Scoring Models

Length prediction differs from scoring models in several ways:

1. **Task Type**: Regression (predict continuous value) vs. ranking (pairwise comparison)
2. **Loss Function**: MSE vs. margin ranking loss
3. **Data Processing**: Extracts response lengths and standardizes them
4. **Output**: Predicted token count vs. relative scores
5. **Output Activation**: None (unbounded regression) vs. tanh (for scores)
6. **Network Activation**: LeakyReLU vs. ReLU
7. **Metrics**: Custom regression metrics vs. pairwise accuracy

### Data Format

**Key advantage**: Uses the same `TrainingData` and `InputData` as scoring models, enabling:
- Easy reuse of existing data pipelines
- Shared preprocessing logic
- Consistent model interfaces
- Training on the same datasets

### Token Length Estimation

Currently uses a simple heuristic for token estimation:
- **Formula**: `word_count * 1.3`
- **Rationale**: Most English words are 1 token, but some are split into multiple tokens
- **Advantages**: Fast, no external dependencies
- **Accuracy**: Sufficient for relative length prediction
- **Alternative**: Could be replaced with tiktoken for exact counts if needed

### Length Scaling and Per-Model Baseline

Lengths are processed in log-space for training, then converted back to raw token counts for output.

A single `_scaler` (fitted on raw lengths by the preprocessor) is reused for log-space training.

**Per-model average (residual learning):**

Both models compute `_model_avg_lengths` — a dict mapping each model name to its mean scaled log-length across the training split. The network/trees learn to predict the *deviation* from this baseline:

- **Training flow**: `raw_length → log → scaler.transform → subtract model_avg → residual` (target)
- **Inference flow**: `model_output (residual) → add model_avg → scaler.inverse_transform → exp → raw_length`

For unseen models at inference time, `model_avg` defaults to `0.0` (global mean in scaled space).

**Benefits:**
- Separates the easily-captured per-model verbosity level from the harder prompt-driven variance
- The network only needs to model relative deviations, reducing the effective prediction range
- Log-space MSE approximates relative (percentage) error, which is more appropriate for lengths
- Better handling of long-tail distributions (very long or very short responses)

### Performance Considerations

- Prediction is fast (batched forward passes)
- Model size depends on `hidden_dims` and embedding dimensions
- Preprocessing is cached for repeated use of same dataset

## Future Enhancements

Potential improvements to the length prediction system:

1. **Better tokenization**: Use tiktoken for exact token counts (currently uses `word_count * 1.3`)
2. **Advanced architectures**: Try transformers, attention mechanisms
3. **Multi-task learning**: Joint training with scoring models
4. **Uncertainty estimation**: Predict confidence intervals, not just point estimates
5. **Category-aware models**: Condition predictions on prompt categories
6. **Length distributions**: Predict full distributions instead of means
7. **Per-model features**: Add model-specific metadata as features
8. **Custom loss functions**: Weighted MSE based on length ranges
9. **Better activations**: Experiment with different activation functions
10. **Ensemble methods**: Combine multiple models for better predictions
