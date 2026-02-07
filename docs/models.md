# Models

This directory contains documentation for different model implementations.

## Implemented Models

### Dense Network Model
- **File**: [dense_network_model.md](models/dense_network_model.md)
- **Type**: Feedforward neural network with learned model embeddings
- **Input**: Fixed-size prompt embeddings + 45 prompt features + Model IDs
- **Output**: Scores in [-1, 1] for each (prompt, model) combination
- **Training**: Margin ranking loss on pairwise comparisons
- **Inference**: Can score any number of models simultaneously

### Simple Scoring Model
- **File**: [simple_scoring_model.md](models/simple_scoring_model.md)
- **Type**: Simple baseline with one learnable score per model (prompt-agnostic)
- **Input**: Model IDs only (ignores prompts)
- **Output**: Same scores for all prompts (based on learned model scores)
- **Training**: Neural network with margin ranking loss + tie/both_bad penalties
- **Inference**: Very fast (lookup table)

### ELO Scoring Model
- **File**: [elo_scoring_model.md](models/elo_scoring_model.md)
- **Type**: ELO rating system baseline (prompt-agnostic)
- **Input**: Model IDs only (ignores prompts)
- **Output**: Same scores for all prompts (based on ELO ratings)
- **Training**: Iterative ELO rating updates (no neural network)
- **Inference**: Very fast (lookup table)


## Response Length Prediction

In addition to scoring models, we also provide models for predicting response lengths.

### Dense Network Length Prediction Model
- **File**: [length_prediction.md](length_prediction.md)
- **Type**: Feedforward neural network for regression
- **Input**: Prompt embeddings + 45 prompt features + Model embeddings (from scoring models)
- **Output**: Predicted response length in tokens for each (prompt, model) combination
- **Training**: Mean Squared Error (MSE) loss on actual response lengths
- **Inference**: Fast batched predictions

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
- `dn_embedding_length_prediction` - Length prediction models with embeddings

### Implementation Details

The embedding loading mechanism uses:
- **`HasEmbeddingModel` protocol**: Defines the interface for models that contain embedding models
- **`load_embedding_model_from_model()` function**: Centralized loading function in `src/models/model_loading.py`
- **Standard model loading**: Uses the same `load_model()` infrastructure as other model loading operations

The protocol ensures type safety and allows for extracting embedding models from any compatible model type without hardcoding dependencies.