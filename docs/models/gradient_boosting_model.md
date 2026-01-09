# Gradient Boosting Model

## Overview

The Gradient Boosting Model uses XGBoost to predict which model is best suited for a given prompt. Unlike neural network models, this model uses tree-based gradient boosting to make predictions.

## Architecture

The model consists of two main components:

1. **Embedding Model**: Trains embeddings for models and prompts (can be frozen or trainable)
2. **XGBoost Classifier**: Binary classifier that predicts which model wins for a given prompt

## Training Process

Training happens in two phases:

### Phase 1: Embedding Training
- Trains the embedding model to create vector representations for:
  - Prompts (using sentence transformers)
  - Models (learned embeddings)
  - Prompt features (e.g., length, complexity)

### Phase 2: Gradient Boosting
- **Incremental Training**: Trees are added one at a time, allowing validation between iterations
- Each "epoch" adds a single tree to the ensemble
- Unlike neural networks, this is not iterating over the data multiple times
- Each tree is fitted once on the data to correct errors from previous trees

## Data Format

### Training Data Preparation
For each comparison pair (model A vs model B), we create two training samples:

1. **Sample for Model A**:
   - Features: `[prompt_embedding, prompt_features, model_embedding_a]`
   - Label: `1.0` if model A wins, `0.0` otherwise

2. **Sample for Model B**:
   - Features: `[prompt_embedding, prompt_features, model_embedding_b]`
   - Label: `1.0` if model B wins, `0.0` otherwise

Samples are ordered as pairs: `[a_1, b_1, a_2, b_2, ...]` so the custom objective function can process them together.

### Custom Objective Function
Instead of standard binary classification, we use a **custom margin ranking objective** similar to PyTorch's `MarginRankingLoss`. This ensures the model learns relative scores between models rather than absolute probabilities.

The loss encourages the winner's predicted score to be higher than the loser's score by at least a margin:
```
loss = max(0, -direction * (score_a - score_b) + margin)
```

This approach allows:
- **Pairwise training**: Models are trained on comparisons
- **Independent inference**: Each model gets scored independently at prediction time
- **Meaningful relative scores**: Strong models competing with each other still get meaningful scores

### Feature Vector
The input to XGBoost is a concatenated feature vector:
```
[prompt_embedding (dim: N), prompt_features (dim: M), model_embedding (dim: K), prompt_categories (dim: C)]
```

Note: `prompt_embedding` and `prompt_categories` are optional and controlled by constructor parameters:
- `use_prompt_embeddings`: Include prompt embeddings in features (default: True)
- `use_prompt_categories`: Include prompt categories in features (default: False)

### Prompt Categories
When `use_prompt_categories=True`, the model incorporates categorical information about prompts into the feature vector:
- **Structure**: 11-dimensional vector encoding prompt tags
  - Creative writing (1 dim): Whether the prompt involves creative writing
  - Criteria (7 dims): complexity, creativity, domain_knowledge, problem_solving, real_world, specificity, technical_accuracy
  - Instruction-following (2 dims): IF tag boolean and score
  - Math (1 dim): Whether the prompt involves math

- **Missing Categories**: If a training entry lacks category tags, a zero vector is used
- **Warning**: The model warns if `use_prompt_categories=True` but some training entries are missing categories
- **Inference**: During inference, category features are set to zero vectors (categories are not computed for inference prompts)

## Sample Balancing

The model supports sample balancing to handle imbalanced model representation:
- Computes inverse frequency weights for each model (inverse of frequency, normalized)
- Applies weights in the custom objective function by multiplying gradients and hessians
- Ensures rare models get adequate representation during training
- Both samples in a pair get weights based on their respective model frequencies

## Hyperparameters

### XGBoost-specific parameters:
- `max_depth`: Maximum tree depth (default: 6)
- `learning_rate`: Step size shrinkage (default: 0.1)
- `colsample_bytree`: Fraction of features used per tree (default: 1.0)
- `reg_alpha`: L1 regularization (default: 0.0)
- `reg_lambda`: L2 regularization (default: 1.0)

**Note:** `subsample` (row sampling) is fixed at 1.0 to preserve the pairwise data structure required by the custom objective function.

### Other parameters:
- `epochs`: Number of boosting rounds (trees to add)
- `embedding_model_epochs`: Epochs for embedding model training
- `balance_model_samples`: Whether to apply sample balancing via sample weights
- `use_prompt_embeddings`: Include prompt embeddings in features (default: True)
- `use_prompt_categories`: Include prompt category features in training (default: False)

## Prediction

For inference, the model:
1. Embeds the input prompt
2. For each candidate model:
   - Creates feature vector: `[prompt_embedding, prompt_features, model_embedding]`
   - Predicts probability that this model is the winner
3. Returns probability scores for all models

## Metrics

Training tracks:
- **Loss**: Margin ranking loss (computed as `max(0, -direction * (pred_a - pred_b) + margin)`)
- **Accuracy**: Pairwise accuracy - percentage of pairs where the model correctly predicts which model should score higher

Both train and validation metrics are computed after each boosting round.

The pairwise accuracy is computed by comparing predictions for both models in each pair and checking if the relative ordering is correct, regardless of the absolute score values.

## Saving and Loading

The model serializes:
- XGBoost model (saved as JSON format)
- Embedding model state
- All hyperparameters and configuration
- Training history

## Key Differences from Neural Network Models

1. **No backpropagation**: Trees are fitted directly using gradient boosting with custom gradients/hessians
2. **No mini-batches**: XGBoost uses the full dataset for each tree
3. **No epochs in traditional sense**: Each iteration adds a new tree, not refining existing parameters
4. **Single pass per tree**: Each tree sees the data once during fitting
5. **No GPU acceleration** (by default): XGBoost uses CPU-based tree algorithms
6. **Custom objective**: Uses margin ranking loss (similar to neural network's `MarginRankingLoss`) but implemented via gradient/hessian computation

## Advantages

- Often requires less hyperparameter tuning than neural networks
- Naturally handles feature interactions through tree splits
- Robust to feature scaling
- Built-in handling of missing values
- Generally faster training for tabular data

## Usage Example

```python
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.embedding_specs.frozen_embedding_spec import FrozenEmbeddingSpec

model = GradientBoostingModel(
    embedding_model_name="all-MiniLM-L6-v2",
    embedding_spec=FrozenEmbeddingSpec(),
    max_depth=6,
    learning_rate=0.1,
    epochs=100,  # Number of trees
    use_prompt_categories=True,  # Enable category features (optional)
)

model.train(data, validation_split=validation_split, epochs=100)
predictions = model.predict(test_data)
```

