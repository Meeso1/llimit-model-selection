# Models

This directory contains documentation for different model implementations.

## Implemented Models

### Prompt-invariant

#### Simple Scoring Model
- **File**: [simple_scoring_model.md](models/simple_scoring_model.md)
- **Type**: Simple baseline with one learnable score per model (prompt-agnostic)
- **Input**: Model IDs only (ignores prompts)
- **Output**: Same scores for all prompts (based on learned model scores)
- **Training**: Neural network with margin ranking loss + tie/both_bad penalties
- **Inference**: Very fast (lookup table)

#### ELO Scoring Model
- **File**: [elo_scoring_model.md](models/elo_scoring_model.md)
- **Type**: ELO rating system baseline (prompt-agnostic)
- **Input**: Model IDs only (ignores prompts)
- **Output**: Same scores for all prompts (based on ELO ratings)
- **Training**: Iterative ELO rating updates (no neural network)
- **Inference**: Very fast (lookup table)

#### Greedy Ranking Model
- **File**: [greedy_ranking_model.md](models/greedy_ranking_model.md)
- **Type**: Non-iterative ranking via greedy net-wins algorithm (prompt-agnostic)
- **Input**: Model IDs only
- **Output**: Rank-derived scores
- **Training**: Single-pass (non-iterative)

#### MCMF Scoring Model (L1 criterion minimization)
- **File**: [mcmf_scoring_model.md](models/mcmf_scoring_model.md)
- **Type**: Min-Cost-Max-Flow graph optimization (prompt-agnostic)
- **Input**: Model IDs only
- **Output**: Flow-derived scores
- **Training**: Single-pass optimization (non-iterative)

#### Least Squares Scoring Model (L2 criterion minimization)
- **File**: [least_squares_scoring_model.md](models/least_squares_scoring_model.md)
- **Type**: Closed-form least-squares fitting of win-rate differences (prompt-agnostic)
- **Input**: Model IDs only
- **Output**: Scores minimizing squared error to pairwise win rates
- **Training**: Single-pass (non-iterative)

### Prompt-based

#### DN Embedding Model
- **File**: [dn_embedding_model.md](models/dn_embedding_model.md)
- **Type**: Dense network with pre-trained model embeddings
- **Input**: Prompt embeddings + 45 prompt features + Model embeddings
- **Output**: Scores in [-1, 1] for each (prompt, model) combination
- **Training**: Margin ranking or Bradley-Terry loss; optionally residual on a base model
- **Inference**: Fast batched predictions

#### Gradient Boosting Model
- **File**: [gradient_boosting_model.md](models/gradient_boosting_model.md)
- **Type**: XGBoost ensemble with learned model embeddings
- **Input**: Prompt embeddings + 45 prompt features + Model embeddings
- **Output**: Scores for each (prompt, model) combination
- **Training**: Incremental tree boosting with configurable pairwise ranking loss

#### Transformer Embedding Model
- **File**: [transformer_embedding_model.md](models/transformer_embedding_model.md)
- **Type**: Fine-tuned transformer with learned model embeddings
- **Input**: Prompt text (tokenized) + 45 prompt features + Model embeddings
- **Output**: Scores in [-1, 1] for each (prompt, model) combination
- **Training**: Fine-tunes transformer with LoRA/QLoRA, margin ranking loss
- **Inference**: Slower than frozen embeddings but potentially more accurate

#### Response Predictive Model
- **File**: [response_predictive_model.md](models/response_predictive_model.md)
- **Type**: Three-component model with explicit response prediction (experimental)
- **Input**: Prompt embeddings + 45 prompt features + Model embeddings + (during training) Response embeddings + 32 response features
- **Output**: Scores in [-1, 1] for each (prompt, model) combination
- **Training**: Joint training with dual-path scoring (real + predicted representations), prediction losses, and distribution-matching regularisation
- **Inference**: Uses predicted response representations (no actual responses needed)
- **Key Innovation**: Dense response-level supervision instead of just binary comparison labels

## Response Length Prediction

In addition to scoring models, we also provide models for predicting response lengths. See [length_prediction.md](length_prediction.md) for full details.

### Dense Network Length Prediction Model (`dn_embedding_length_prediction`)
- **Type**: Residual feedforward network with input projection layers
- **Input**: Prompt embeddings + 45 prompt features + Model embeddings + learned model ID embedding
- **Output**: Predicted response length in tokens for each (prompt, model) combination
- **Training**: MSE loss on standardized log-lengths, learns residuals relative to per-model average

### Gradient Boosting Length Prediction Model (`gb_length_prediction`)
- **Type**: XGBoost ensemble for regression
- **Input**: Configurable — any subset of prompt embeddings, prompt features, model embeddings, one-hot model ID
- **Output**: Predicted response length in tokens for each (prompt, model) combination
- **Training**: XGBoost with MSE objective on standardized log-length residuals; best-epoch tracking

### Simple Length Prediction Model (`simple_length_prediction`)
- **File**: [simple_length_prediction_model.md](models/simple_length_prediction_model.md)
- **Type**: Per-model OLS linear regression (non-iterative baseline)
- **Input**: Configurable subset of the 45 prompt features (or intercept-only)
- **Output**: Predicted response length in tokens
- **Training**: Single-pass closed-form OLS; no model embeddings required

## Embedding Model Reuse

Many models (e.g., `DnEmbeddingModel`, `TransformerEmbeddingModel`, `GradientBoostingModel`, `DnEmbeddingLengthPredictionModel`) learn embeddings for LLM models as part of their training. These embeddings can be reused across models to:
- Share learned model representations
- Bootstrap new models with pre-trained embeddings
- Avoid retraining embeddings from scratch

### Loading Embedding Models

To load embedding models from a previously trained model, use the `load_embedding_model_from` parameter:

```python
# Format: "model_type/model_name"
new_model = DnEmbeddingModel(
    load_embedding_model_from="dn_embedding/my_base_model",
    # ... other parameters
)
```

Supported source model types:
- `dn_embedding` - Dense network models with embeddings
- `transformer_embedding` - Transformer-based models with embeddings
- `gradient_boosting` - Gradient boosting models with embeddings
- `dn_embedding_length_prediction` - Dense network length prediction models with embeddings
- `gb_length_prediction` - Gradient boosting length prediction models with embeddings

### Implementation Details

The embedding loading mechanism uses:
- **`HasEmbeddingModel` protocol**: Defines the interface for models that contain embedding models
- **`load_embedding_model_from_model()` function**: Centralized loading function in `src/models/model_loading.py`
- **Standard model loading**: Uses the same `load_model()` infrastructure as other model loading operations

The protocol ensures type safety and allows for extracting embedding models from any compatible model type without hardcoding dependencies.