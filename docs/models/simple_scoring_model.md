# Simple Scoring Model

## Overview

The `SimpleScoringModel` is a baseline model that learns a single score for each LLM without looking at prompts or responses. It serves as a simple baseline for comparison with more sophisticated models.

## Architecture

The model is extremely simple:
- **Network**: A single learnable parameter (score) for each model
- **No prompt embeddings**: Completely ignores prompt content
- **No response embeddings**: Completely ignores response content
- **Preprocessor**: `SimpleScoringPreprocessor` extracts model comparisons and encodes model names

This is essentially a lookup table where each LLM has a learned score based on pairwise comparisons.

## Comparison Types

The model handles four types of comparisons:

1. **model_a_wins / model_b_wins**: Standard pairwise ranking using `MarginRankingLoss`
   - Loss encourages winner's score to be higher than loser's
   - Accuracy: correct if winner has higher score than loser
   
2. **tie**: Both models performed well
   - Loss encourages both models to have scores above `tie_both_bad_epsilon`
   - Uses `ReLU(-score + epsilon)` to penalize scores below threshold
   - Weighted by `non_ranking_loss_coeff` (default: 0.01)
   - Accuracy: correct if both scores are positive
   
3. **both_bad**: Both models performed poorly
   - Loss encourages both models to have scores below `-tie_both_bad_epsilon`
   - Uses `ReLU(score + epsilon)` to penalize scores above threshold
   - Weighted by `non_ranking_loss_coeff` (default: 0.01)
   - Accuracy: correct if both scores are negative

The overall accuracy is computed across all comparison types, giving equal weight to each comparison.

### Loss Weighting

- **Ranking loss**: Weight = 1.0 (full weight)
- **Tie/both_bad loss**: Weight = `non_ranking_loss_coeff` (default: 0.01)

This weighting ensures that the primary objective is to learn the ranking between models, while tie and both_bad comparisons provide secondary guidance for absolute score calibration.

## Training

### Preprocessing
1. **SimpleScoringPreprocessor** extracts all comparisons from training data
2. Creates a `StringEncoder` to map model names to integer IDs
3. Returns `PreprocessedTrainingData` with encoded comparisons

### Training Process
1. **Initialization**: Preprocesses data and initializes network with one parameter per model
2. **Data Split**: Optionally splits preprocessed data into train/validation sets
3. **Training**: Uses custom loss function that handles all comparison types
   - Pairwise ranking for wins/losses
   - Positive score targets for ties
   - Negative score targets for both_bad

### Sample Balancing
If `balance_model_samples=True`, the model uses `WeightedRandomSampler` to ensure rare models get adequate representation during training.

### Training History
The model tracks enhanced training history with `TrainingHistoryEntry`:
- Standard metrics: loss, accuracy, validation loss/accuracy
- **Additional metrics** (stored in `additional_metrics` dict):
  - **Score statistics**:
    - `avg_score`: Average score across all models
    - `top_10_pct_score`: Average score of top 10% models
    - `bottom_10_pct_score`: Average score of bottom 10% models
  - **Training loss components**:
    - `ranking_loss`: Loss from pairwise ranking comparisons
    - `tie_loss`: Loss from tie comparisons
    - `both_bad_loss`: Loss from both_bad comparisons
  - **Training accuracy components**:
    - `ranking_accuracy`: Accuracy on pairwise ranking (winner/loser)
    - `tie_accuracy`: Accuracy on ties (both models positive)
    - `both_bad_accuracy`: Accuracy on both_bad (both models negative)
  - **Validation loss components** (if validation is enabled):
    - `val_ranking_loss`: Validation loss from pairwise ranking
    - `val_tie_loss`: Validation loss from ties
    - `val_both_bad_loss`: Validation loss from both_bad
  - **Validation accuracy components** (if validation is enabled):
    - `val_ranking_accuracy`: Validation accuracy on ranking
    - `val_tie_accuracy`: Validation accuracy on ties
    - `val_both_bad_accuracy`: Validation accuracy on both_bad

## Prediction

During prediction, the model:
1. Returns the same ranking of all models for every prompt (since it doesn't look at prompts)
2. The ranking is based purely on the learned scores
3. Unknown models (not seen during training) receive a score of 0

This makes predictions very fast but unable to adapt to different prompt types.

## API

### Initialization
```python
from src.models.simple_scoring_model import SimpleScoringModel
from src.utils.data_split import ValidationSplit

model = SimpleScoringModel(
    optimizer_spec=None,  # Default: AdamWSpec()
    balance_model_samples=True,
    print_every=1,
    tie_both_bad_epsilon=1e-2,  # Threshold for tie/both_bad score targets
    non_ranking_loss_coeff=0.01,  # Weight for tie/both_bad loss relative to ranking loss
    wandb_details=None,
)
```

### Training
```python
model.train(
    data=training_data,  # TrainingData with EvaluationEntry objects
    validation_split=ValidationSplit(val_fraction=0.2, seed=42),
    epochs=100,
    batch_size=32,
)
```

### Prediction
```python
from src.data_models.data_models import InputData

input_data = InputData(
    prompts=["prompt1", "prompt2"],
    model_names=["model_1", "model_2", "model_3"],
)

output = model.predict(input_data)
# output.scores: dict[str, np.ndarray] - model_name -> scores [n_prompts]
```

### Getting Model Scores
```python
# Get scores for all known models
all_scores = model.get_all_model_scores()
# Returns: dict[str, float] mapping model names to scores

for model_name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: {score:.4f}")
```

## Parameters

### Constructor
- `optimizer_spec`: Optimizer specification (default: AdamWSpec)
- `balance_model_samples`: Whether to balance samples by model frequency (default: True)
- `print_every`: How often to print epoch results (default: 1)
- `tie_both_bad_epsilon`: Threshold for tie/both_bad comparisons (default: 1e-2)
  - For ties: both scores should be > epsilon
  - For both_bad: both scores should be < -epsilon
  - Larger values create stronger separation between good/bad models
- `non_ranking_loss_coeff`: Weight for tie/both_bad loss relative to ranking loss (default: 0.01)
  - Controls the importance of tie/both_bad comparisons vs ranking comparisons
  - Lower values focus more on ranking accuracy
  - Higher values focus more on absolute score calibration
- `wandb_details`: Weights & Biases configuration (default: None)

### Train method
- `data`: TrainingData with pairwise comparisons
- `validation_split`: Fraction of data for validation (0-1), or None
- `epochs`: Number of training epochs
- `batch_size`: Batch size

## State Persistence

The model can be saved and loaded:

```python
# Save
state_dict = model.get_state_dict()

# Load
model = SimpleScoringModel.load_state_dict(state_dict)
```

The state includes:
- Model scores (learned parameters)
- Model encoder (mapping from model names to IDs)
- Training history (including score statistics)
- Optimizer configuration
- Model configuration

## Training Output Example

```
Epoch    5: loss = 0.0977, accuracy = 50.00%, scores: avg=0.001, top10%=0.006, bot10%=-0.002 - 0.00s
Epoch   10: loss = 0.0472, accuracy = 75.00%, scores: avg=0.003, top10%=0.014, bot10%=-0.003 - 0.00s
```

With validation:
```
Epoch    2: loss = 0.0751/0.0960, accuracy = 75.00%/66.67%, scores: avg=0.002, top10%=0.006, bot10%=-0.000 - 0.00s
```

The accuracy now reflects performance across all comparison types (ranking, ties, and both_bad).

## Accessing Detailed Metrics

```python
# Get training history with detailed metrics
history = model.get_history()

# Access standard metrics
print(history.total_loss)  # list[float]
print(history.train_accuracy)  # list[float]
print(history.val_loss)  # list[float | None]
print(history.val_accuracy)  # list[float | None]

# Access training metrics
print(history.additional_metrics["ranking_accuracy"])  # list[float | None]
print(history.additional_metrics["tie_accuracy"])  # list[float | None]
print(history.additional_metrics["both_bad_accuracy"])  # list[float | None]
print(history.additional_metrics["ranking_loss"])  # list[float | None]
print(history.additional_metrics["tie_loss"])  # list[float | None]
print(history.additional_metrics["both_bad_loss"])  # list[float | None]

# Access validation metrics (if validation was used)
print(history.additional_metrics["val_ranking_accuracy"])  # list[float | None]
print(history.additional_metrics["val_tie_accuracy"])  # list[float | None]
print(history.additional_metrics["val_both_bad_accuracy"])  # list[float | None]
print(history.additional_metrics["val_ranking_loss"])  # list[float | None]
print(history.additional_metrics["val_tie_loss"])  # list[float | None]
print(history.additional_metrics["val_both_bad_loss"])  # list[float | None]

# Access score statistics
print(history.additional_metrics["avg_score"])  # list[float | None]
print(history.additional_metrics["top_10_pct_score"])  # list[float | None]
print(history.additional_metrics["bottom_10_pct_score"])  # list[float | None]
```

## Use Case

This model is primarily useful as a **baseline** to compare against more sophisticated models that consider prompt content. It answers the question: "How much better can we do by actually looking at the prompts?"

**Interpretation of results:**
- If this simple model performs well, it suggests that some models are generally better than others across all prompt types
- If it performs poorly compared to prompt-aware models, it indicates that prompt content matters significantly for model selection
- The score statistics (avg, top 10%, bottom 10%) help understand the spread of model quality

**Advantages:**
- Extremely fast training and inference
- No need for embedding models
- Easy to interpret (just a score per model)
- Handles ties and both_bad comparisons

**Limitations:**
- Cannot adapt recommendations to different prompt types
- Assumes model performance is consistent across all prompts
- May underperform when prompt type significantly affects model selection

