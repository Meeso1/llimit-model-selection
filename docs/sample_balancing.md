# Sample Balancing

## Overview

Sample balancing addresses the issue where some models appear much more frequently in the training data than others. Without balancing, the network may learn to score frequently-occurring models better simply due to more training exposure, rather than actual performance.

## Implementation

The `DenseNetworkModel` uses weighted random sampling to balance model representation during training. This can be enabled/disabled with the `balance_model_samples` parameter (default: `True`).

### How It Works

1. **Count model frequencies**: For each training pair, both models involved are counted
2. **Compute weights**: Each model gets weight = 1 / frequency (inverse frequency weighting)
3. **Assign pair weights**: Each training pair gets the weight of its rarest model
4. **Sample with replacement**: Pairs containing rare models are sampled more frequently

This ensures that rare models appear in training batches with similar frequency as common models.

### Example

If the data contains:
- Model A: 1000 pairs
- Model B: 100 pairs
- Model C: 10 pairs

Without balancing, Model A dominates training. With balancing:
- Pairs with Model C are sampled ~100x more frequently
- Pairs with Model B are sampled ~10x more frequently
- All models get roughly equal representation

## Usage

```python
from src.models.dense_network_model import DenseNetworkModel

# Enable balancing (default)
model = DenseNetworkModel(
    balance_model_samples=True,
)

# Disable balancing
model = DenseNetworkModel(
    balance_model_samples=False,
)
```

## When to Use

**Enable balancing when:**
- Training data has significant model imbalance
- You want fair comparison across all models
- Rare models are important for your use case

**Disable balancing when:**
- Data is already balanced
- You want training to reflect real-world frequencies
- Computational cost is a concern (balanced sampling uses replacement, so more unique samples per epoch)

## Limitations

- Does not handle extreme cases (e.g., single occurrence of a model)
- Uses sampling with replacement, so same pairs may appear multiple times per epoch
- Only applied to training data, not validation data
- Balancing is per-model, not per-prompt or per-prompt-model combination

## Implementation Details

Located in `DenseNetworkModel._create_balanced_sampler()` method. Uses PyTorch's `WeightedRandomSampler` with `replacement=True`.

The sampler is created during dataloader preparation and only applied when `is_training=True`.

