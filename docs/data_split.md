# Train/Validation Data Splitting

## Overview

The `data_split` module provides a simple utility to split training data into train and validation sets.

## Function

### `train_val_split`

Splits a `TrainingData` object into train and validation sets deterministically.

**Parameters:**
- `data: TrainingData` - The training data to split
- `val_fraction: float` - Fraction of data to use for validation (default: 0.2)
- `seed: int` - Random seed for reproducibility (default: 42)

**Returns:**
- `tuple[TrainingData, TrainingData]` - Train and validation data

**Behavior:**
- Uses numpy's RandomState with a fixed seed for deterministic shuffling
- Shuffles all entries randomly, then splits by the specified fraction
- Validation set is taken from the beginning of the shuffled indices
- Raises ValueError if val_fraction is not in (0, 1) or if the result would be an empty set

## Usage Example

```python
from src.data_loading import load_training_data
from src.utils.data_split import train_val_split

data = load_training_data(...)
train_data, val_data = train_val_split(data, val_fraction=0.2, seed=42)
```

