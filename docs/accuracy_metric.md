# Accuracy Metric

## Overview

The accuracy metric measures how well the model's predictions align with human evaluations in pairwise comparisons.

## Definition

For each pair of models (A, B) with a human preference label:
- **Accuracy** = percentage of pairs where the model correctly predicts which model has the higher score

The prediction is considered correct when:
- Human prefers A (label = 1) AND model assigns `score_a > score_b`
- Human prefers B (label = -1) AND model assigns `score_b > score_a`

Ties (when scores are exactly equal) are treated as incorrect predictions.

## Implementation

### Function: `compute_pairwise_accuracy`

Located in `src/utils/accuracy.py`

**Parameters:**
- `scores_a: torch.Tensor` - Scores for model A, shape `[batch_size]`
- `scores_b: torch.Tensor` - Scores for model B, shape `[batch_size]`
- `labels: torch.Tensor` - Ground truth labels, shape `[batch_size]`
  - `1` if A wins
  - `-1` if B wins

**Returns:**
- `float` - Accuracy value in `[0, 1]`

**Algorithm:**
1. Compute predictions: `sign(score_a - score_b)`
2. Handle ties: treat as incorrect
3. Compare predictions to labels
4. Return fraction of correct predictions

## Training Integration

Both training and validation accuracies are computed and logged:
- Computed per batch and averaged over all batches in an epoch
- Stored in `TrainingHistoryEntry` with fields `train_accuracy` and `val_accuracy`
- Logged to Weights & Biases if enabled

## Visualization

Accuracy is plotted using an inverted logarithmic scale: `-log(1 - accuracy)`

This transformation:
- Maps `[0, 1]` to `[0, ∞)`
- Emphasizes improvements near perfect accuracy (1.0)
- Makes small improvements at high accuracy more visible

For example:
- 50% accuracy → 0.69
- 90% accuracy → 2.30
- 99% accuracy → 4.61
- 99.9% accuracy → 6.91

