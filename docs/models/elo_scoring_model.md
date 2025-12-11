# ELO Scoring Model

## Overview

The `EloScoringModel` is a baseline model that learns ratings for each LLM using the ELO rating system, without looking at prompts or responses. It uses a classic chess-style rating algorithm instead of neural network training, making it interpretable and computationally efficient.

## Architecture

The model uses the ELO rating system:
- **Rating System**: Each model has a single rating value (initialized at `initial_rating`, typically 1500)
- **No neural network**: Updates are purely algorithmic based on ELO formulas
- **No prompt embeddings**: Completely ignores prompt content
- **No response embeddings**: Completely ignores response content
- **Preprocessor**: `SimpleScoringPreprocessor` extracts model comparisons and encodes model names

This is a lookup table where each LLM has a rating that is updated based on comparison outcomes using the ELO algorithm.

## ELO Rating System

### Standard ELO Updates

For pairwise comparisons (model_a_wins / model_b_wins), the algorithm:

1. **Computes expected score** using the logistic function:
   ```
   expected_a = 1 / (1 + 10^((rating_b - rating_a) / 400))
   expected_b = 1 - expected_a
   ```

2. **Determines actual score**:
   - Winner: 1.0
   - Loser: 0.0

3. **Updates ratings**:
   ```
   rating_a += k_factor * (actual_a - expected_a)
   rating_b += k_factor * (actual_b - expected_b)
   ```

Where `k_factor` (default: 32.0) controls the maximum rating change per comparison.

### Extended ELO for Special Cases

The model extends traditional ELO to handle ties and both_bad comparisons:

## Comparison Types

The model handles four types of comparisons:

1. **model_a_wins / model_b_wins**: Standard ELO rating update
   - Winner gets actual score of 1.0, loser gets 0.0
   - Ratings adjusted based on expected vs actual outcomes
   - Accuracy: correct if winner has higher rating than loser
   
2. **tie**: Both models performed well
   - Both models get actual score of 0.5 (standard ELO tie handling)
   - Additional penalty applied if ratings are below `initial_rating + tie_both_bad_epsilon`
   - Penalty increases actual score by `non_ranking_loss_coeff * 0.5` to push ratings up
   - Accuracy: correct if both ratings are above initial_rating (positive scores)
   
3. **both_bad**: Both models performed poorly
   - Both models get actual score of 0.0
   - Additional penalty applied if ratings are above `initial_rating - tie_both_bad_epsilon`
   - Penalty decreases actual score by `non_ranking_loss_coeff * 0.5` to push ratings down
   - Accuracy: correct if both ratings are below initial_rating (negative scores)

The overall accuracy is computed across all comparison types, giving equal weight to each comparison.

### Rating Interpretation

- **Score = rating - initial_rating**: Normalized score centered at 0
- **Positive score**: Model is better than average (starting rating)
- **Negative score**: Model is worse than average
- **Magnitude**: Indicates confidence in rating (larger = more extreme performance)

## Training

### Preprocessing
1. **SimpleScoringPreprocessor** extracts all comparisons from training data
2. Creates a `StringEncoder` to map model names to integer IDs
3. Returns `PreprocessedTrainingData` with encoded comparisons

### Training Process
1. **Initialization**: Preprocesses data and initializes all ratings to `initial_rating`
2. **Data Split**: Optionally splits preprocessed data into train/validation sets
3. **Training Epochs**: Each epoch:
   - Shuffles comparisons (with optional weighted sampling for balance)
   - Processes each comparison sequentially, updating ratings
   - Applies ELO formula with special handling for ties and both_bad
4. **Convergence**: Ratings converge as the same comparisons are revisited across epochs

### Sample Balancing
If `balance_model_samples=True`, the model uses weighted sampling to ensure rare models get adequate representation during training. Weights are computed based on inverse frequency of model appearances.

### Training History
The model tracks training history with `TrainingHistoryEntry`:
- Standard metrics: accuracy, validation accuracy (no loss, as ELO is not loss-based)
- **Additional metrics** (stored in `additional_metrics` dict):
  - **Rating statistics**:
    - `avg_rating`: Average score across all models (relative to initial_rating)
    - `top_10_pct_rating`: Average score of top 10% models
    - `bottom_10_pct_rating`: Average score of bottom 10% models
    - `avg_rating_change`: Average absolute rating change per comparison in this epoch
  - **Training accuracy components**:
    - `ranking_accuracy`: Accuracy on pairwise ranking (winner/loser)
    - `tie_accuracy`: Accuracy on ties (both models positive)
    - `both_bad_accuracy`: Accuracy on both_bad (both models negative)
  - **Validation accuracy components** (if validation is enabled):
    - `val_ranking_accuracy`: Validation accuracy on ranking
    - `val_tie_accuracy`: Validation accuracy on ties
    - `val_both_bad_accuracy`: Validation accuracy on both_bad

## Prediction

During prediction, the model:
1. Returns the same ranking of all models for every prompt (since it doesn't look at prompts)
2. The ranking is based purely on the learned ELO ratings
3. Ratings are converted to scores by subtracting `initial_rating` (centering around 0)
4. Unknown models (not seen during training) receive a score of 0 (equivalent to initial_rating)

This makes predictions very fast but unable to adapt to different prompt types.

## API

### Initialization
```python
from src.models.elo_scoring_model import EloScoringModel
from src.utils.data_split import ValidationSplit

model = EloScoringModel(
    initial_rating=1500.0,  # Starting rating for all models
    k_factor=32.0,  # Maximum rating change per comparison
    balance_model_samples=True,
    print_every=1,
    tie_both_bad_epsilon=100.0,  # Threshold for tie/both_bad penalties
    non_ranking_loss_coeff=0.1,  # Weight for tie/both_bad penalties
    wandb_details=None,
)
```

### Training
```python
model.train(
    data=training_data,  # TrainingData with EvaluationEntry objects
    validation_split=ValidationSplit(val_fraction=0.2, seed=42),
    epochs=10,  # Number of passes through the data
    batch_size=32,  # Not used in ELO (kept for API consistency)
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
# Returns: dict[str, float] mapping model names to scores (relative to initial_rating)

for model_name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: {score:.1f}")
```

## Parameters

### Constructor
- `initial_rating`: Starting ELO rating for all models (default: 1500.0)
  - Standard value from chess ratings
  - Higher values give more "headroom" for ratings to increase
- `k_factor`: Maximum rating change per comparison (default: 32.0)
  - Higher values = faster convergence but more volatility
  - Lower values = slower convergence but more stability
  - Standard values: 32 (active players), 16 (established players), 10 (masters)
- `balance_model_samples`: Whether to balance samples by model frequency (default: True)
  - Ensures rare models get adequate training
- `print_every`: How often to print epoch results (default: 1)
- `tie_both_bad_epsilon`: Threshold for tie/both_bad comparisons (default: 100.0)
  - For ties: both ratings should be > initial_rating + epsilon
  - For both_bad: both ratings should be < initial_rating - epsilon
  - Larger values create stronger separation between good/bad models
  - Scale matches typical ELO rating differences
- `non_ranking_loss_coeff`: Weight for tie/both_bad penalties (default: 0.1)
  - Controls the importance of tie/both_bad comparisons vs ranking comparisons
  - Lower values focus more on ranking accuracy
  - Higher values focus more on absolute rating calibration
- `wandb_details`: Weights & Biases configuration (default: None)

### Train method
- `data`: TrainingData with pairwise comparisons
- `validation_split`: Fraction of data for validation (0-1), or None
- `epochs`: Number of passes through the training data
- `batch_size`: Not used in ELO (kept for API consistency)

## State Persistence

The model can be saved and loaded:

```python
# Save
state_dict = model.get_state_dict()

# Load
model = EloScoringModel.load_state_dict(state_dict)
```

The state includes:
- Model ratings (current ELO ratings for all models)
- Model encoder (mapping from model names to IDs)
- Training history (including rating statistics)
- Model configuration

## Training Output Example

```
Epoch    1: accuracy = 72.45%, ratings: avg=0.0, top10%=45.2, bot10%=-42.1, Δrating=12.34 - 0.05s
Epoch    2: accuracy = 75.12%, ratings: avg=0.0, top10%=68.5, bot10%=-65.8, Δrating=8.67 - 0.05s
Epoch    5: accuracy = 78.34%, ratings: avg=0.0, top10%=112.3, bot10%=-108.7, Δrating=3.21 - 0.05s
```

With validation:
```
Epoch    2: accuracy = 75.12%/73.45%, ratings: avg=0.0, top10%=68.5, bot10%=-65.8, Δrating=8.67 - 0.05s
```

The accuracy reflects performance across all comparison types (ranking, ties, and both_bad).

## Accessing Detailed Metrics

```python
# Get training history with detailed metrics
history = model.get_history()

# Access standard metrics
print(history.train_accuracy)  # list[float]
print(history.val_accuracy)  # list[float | None]

# Access training metrics
print(history.additional_metrics["ranking_accuracy"])  # list[float | None]
print(history.additional_metrics["tie_accuracy"])  # list[float | None]
print(history.additional_metrics["both_bad_accuracy"])  # list[float | None]
print(history.additional_metrics["avg_rating_change"])  # list[float | None]

# Access validation metrics (if validation was used)
print(history.additional_metrics["val_ranking_accuracy"])  # list[float | None]
print(history.additional_metrics["val_tie_accuracy"])  # list[float | None]
print(history.additional_metrics["val_both_bad_accuracy"])  # list[float | None]

# Access rating statistics
print(history.additional_metrics["avg_rating"])  # list[float | None]
print(history.additional_metrics["top_10_pct_rating"])  # list[float | None]
print(history.additional_metrics["bottom_10_pct_rating"])  # list[float | None]
```

## Comparison with SimpleScoringModel

### EloScoringModel
- **Algorithm**: ELO rating system (iterative updates)
- **Training**: No neural network, purely algorithmic
- **Convergence**: Requires multiple epochs to stabilize ratings
- **Interpretability**: Ratings follow standard ELO scale (like chess)
- **Speed**: Very fast (no backpropagation)
- **Parameters**: `initial_rating`, `k_factor` control behavior

### SimpleScoringModel
- **Algorithm**: Neural network with margin ranking loss
- **Training**: Backpropagation with PyTorch
- **Convergence**: Can converge in fewer epochs with good optimizer
- **Interpretability**: Scores are abstract (not tied to a standard scale)
- **Speed**: Fast (simple network, but requires GPU/backward pass)
- **Parameters**: `optimizer_spec` controls learning rate and dynamics

Both models:
- Ignore prompts (prompt-agnostic)
- Learn a single score per model
- Handle ties and both_bad comparisons
- Use same preprocessor and data structures

## Use Case

This model is primarily useful as:

1. **Baseline**: Compare against more sophisticated prompt-aware models
2. **Interpretable Alternative**: ELO ratings are familiar and have standard interpretation
3. **Research Tool**: Study how ELO ratings compare to learned neural scores
4. **Fast Prototyping**: No need for GPU or complex optimizer tuning

**Interpretation of results:**
- Ratings follow standard ELO interpretation (1500 = average, higher = better)
- Rating differences predict win probability: 400 points ≈ 10:1 odds
- Stable ratings indicate consistent model performance
- High `avg_rating_change` indicates ratings haven't converged yet

**Advantages:**
- Extremely fast training (no backpropagation)
- Interpretable ratings (standard ELO scale)
- No hyperparameter tuning (k_factor has standard values)
- Handles ties and both_bad comparisons naturally
- No need for GPU

**Limitations:**
- Cannot adapt recommendations to different prompt types
- Assumes model performance is consistent across all prompts
- May require more epochs to converge than neural network
- Less flexible than learned loss functions

## Parameter Tuning Tips

### k_factor
- **Start with 32.0** (standard for active competition)
- **Increase (e.g., 64.0)** if you have few comparisons and want faster convergence
- **Decrease (e.g., 16.0)** if you have many comparisons and want more stability

### initial_rating
- **Keep at 1500.0** for standard ELO interpretation
- Only change if you need a different baseline

### tie_both_bad_epsilon
- **Start with 100.0** (reasonable for ELO scale)
- **Increase** if you want stronger separation between good/bad models
- **Decrease** if penalties are too aggressive

### non_ranking_loss_coeff
- **Start with 0.1** (moderate influence)
- **Increase** if tie/both_bad accuracy is poor
- **Decrease** if ranking accuracy is suffering

### epochs
- **Monitor `avg_rating_change`**: when it plateaus, training has converged
- Typically needs 5-20 epochs depending on k_factor and dataset size

