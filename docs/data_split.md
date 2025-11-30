# Train/Validation Data Splitting

## Overview

The project provides multiple ways to split data for training and validation:

1. **`ValidationSplit` dataclass** (recommended): Configure splitting in `train()` method
2. **`train_val_split()`**: Manual splitting of raw training data
3. **`split_preprocessed_data()`**: Splitting of already-preprocessed data

## Recommended Approach: ValidationSplit

The simplest way to use validation is with the `ValidationSplit` dataclass:

```python
from src.models.dense_network_model import DenseNetworkModel
from src.utils.data_split import ValidationSplit

model = DenseNetworkModel()
model.train(
    data=full_dataset,
    validation_split=ValidationSplit(val_fraction=0.2, seed=42),
    epochs=10,
)
```

**Benefits:**
- Simplest API - pass full dataset to `train()`
- Preprocessing happens once on full dataset
- Split occurs after preprocessing
- Model encoder is fitted on all data

### ValidationSplit dataclass

**Fields:**
- `val_fraction: float` - Fraction for validation (default: 0.2)
- `seed: int` - Random seed for reproducibility (default: 42)

## Manual Splitting Functions

### train_val_split()

Splits raw `TrainingData` before preprocessing:

```python
from src.utils.data_split import train_val_split

train_data, val_data = train_val_split(
    data=full_dataset,
    val_fraction=0.2,
    seed=42,
)
```

**Parameters:**
- `data: TrainingData` - Training data to split
- `val_fraction: float` - Validation fraction (default: 0.2)
- `seed: int` - Random seed (default: 42)

**Returns:** `tuple[TrainingData, TrainingData]`

**Note:** This splits *before* preprocessing, so the model encoder is only fitted on the training portion.

### split_preprocessed_data()

Splits already-preprocessed data:

```python
from src.utils.data_split import split_preprocessed_data

train_preprocessed, val_preprocessed = split_preprocessed_data(
    preprocessed_data=preprocessed,
    val_fraction=0.2,
    seed=42,
)
```

**Parameters:**
- `preprocessed_data: PreprocessedTrainingData` - Preprocessed data to split
- `val_fraction: float` - Validation fraction (default: 0.2)
- `seed: int` - Random seed (default: 42)

**Returns:** `tuple[PreprocessedTrainingData, PreprocessedTrainingData]`

**Use case:** Creating different splits without re-preprocessing.

## Implementation Details

All functions:
- Use NumPy's `RandomState` with fixed seed for deterministic shuffling
- Shuffle indices randomly, then split by specified fraction
- Raise `ValueError` if `val_fraction` not in (0, 1) or result would be empty
- `split_preprocessed_data()` preserves shared model encoder in both splits

