# Scoring Sensitivity Analysis

`src/analysis/scoring_sensitivity.py` provides model-independent interpretability
functions for trained `ScoringModelBase` instances. All functions take raw
`TrainingData` and a `ValidationSplit`; they call `model.evaluate()` or
`model.predict()` through the standard interface without accessing internal state.

## Preprocessor interface

### `SwitchablePreprocessor` (base class)

Defined in `src/preprocessing/switchable_preprocessor.py`. Preprocessors that
inherit from it gain four context managers which temporarily modify how
`preprocess_for_inference()` behaves:

| Context manager | Effect during inference |
|---|---|
| `shuffled_prompts(seed)` | Permutes prompt embeddings and features across the batch |
| `set_prompt_embedding_to_mean()` | Replaces all prompt embeddings with their batch mean |
| `set_prompt_features_to_mean()` | Replaces all prompt feature vectors with their batch mean |
| `permuted_feature(feature_idx, seed)` | Permutes a single feature column across the batch |

The base class owns all switch state (flags and seeds) and the shared
`_apply_prompt_switches(emb, feat)` helper. Concrete preprocessors call
`super().__init__()` and `self._apply_prompt_switches(emb, feat)` inside
`preprocess_for_inference()`.

Currently implemented by:
- `PromptEmbeddingPreprocessor` (`DnEmbeddingModel`, `DenseNetworkModel`)
- `ResponsePredictivePreprocessor` (`ResponsePredictiveModel`)

Note: model embeddings are not managed by the preprocessor and are therefore
not covered by these switches.

## `ScoringModelBase` additions

### `get_preprocessor() -> Any` (abstract)
Each `ScoringModelBase` subclass implements this. Models with a preprocessor
return `self.preprocessor`; models without one raise `NotImplementedError`.

### `evaluate(data, split) -> tuple[float, float]`
Splits raw `TrainingData` at the entry level according to `split`, then calls
`predict()` independently on each subset. Returns `(train_accuracy,
val_accuracy)`.

Entries are grouped by model pair (`frozenset{model_a, model_b}`) so that
`predict()` is called once per unique pair with only those two models —
avoiding scoring the full pool of ~50 models for every comparison.

## Analysis functions

All functions return a `float` representing the val-set accuracy drop
(`baseline_val - modified_val`), except `compute_ranking_consistency`.

### `compute_prompt_sensitivity`
```python
def compute_prompt_sensitivity(
    model: ScoringModelBase,
    data: TrainingData,
    split: ValidationSplit,
    n_repeats: int = 5,
    seed: int = 42,
) -> float
```
Shuffles prompt embeddings and features n_repeats times and returns the mean
val accuracy drop. Requires `SwitchablePreprocessor`.

### `compute_prompt_embedding_sensitivity`
```python
def compute_prompt_embedding_sensitivity(
    model: ScoringModelBase,
    data: TrainingData,
    split: ValidationSplit,
) -> float
```
Replaces all prompt embeddings with their batch mean and returns the val
accuracy drop. Requires `SwitchablePreprocessor`.

### `compute_prompt_features_sensitivity`
```python
def compute_prompt_features_sensitivity(
    model: ScoringModelBase,
    data: TrainingData,
    split: ValidationSplit,
) -> float
```
Replaces all prompt feature vectors with their batch mean and returns the val
accuracy drop. Requires `SwitchablePreprocessor`.

### `compute_feature_importance`
```python
def compute_feature_importance(
    model: ScoringModelBase,
    data: TrainingData,
    split: ValidationSplit,
    seed: int = 42,
) -> dict[str, float]
```
Permutation feature importance over all prompt features. Returns a dict
mapping feature name to val accuracy drop. Feature names come from
`get_prompt_feature_names()`, which matches the order produced by
`extract_and_transform_all_prompt_features`. Requires `SwitchablePreprocessor`.

### `compute_ranking_consistency`
```python
def compute_ranking_consistency(
    model: ScoringModelBase,
    data: TrainingData,
    split: ValidationSplit,
    n_splits: int = 10,
    split_fraction: float = 0.5,
    seed: int = 42,
) -> float
```
Evaluates how stable model rankings are across random subsets of the val
prompts. Returns mean pairwise Spearman rank correlation. Values near 1.0
indicate stable rankings. Does **not** require `SwitchablePreprocessor`.

## Usage example

```python
from src.analysis.scoring_sensitivity import (
    compute_prompt_sensitivity,
    compute_prompt_embedding_sensitivity,
    compute_prompt_features_sensitivity,
    compute_feature_importance,
    compute_ranking_consistency,
)
from src.utils.data_split import ValidationSplit

split = ValidationSplit(val_fraction=0.2, seed=42)

# model is a trained ScoringModelBase; data is TrainingData
sensitivity   = compute_prompt_sensitivity(model, data, split)
emb_drop      = compute_prompt_embedding_sensitivity(model, data, split)
feat_drop     = compute_prompt_features_sensitivity(model, data, split)
importances   = compute_feature_importance(model, data, split)
consistency   = compute_ranking_consistency(model, data, split)

train_acc, val_acc = model.evaluate(data, split)
```

## Error handling

- `get_preprocessor()` raises `NotImplementedError` for models without a
  preprocessor (`LeastSquaresScoringModel`, `McMfScoringModel`).
- Switch-dependent functions call `_require_switchable()`, which raises
  `TypeError` if the preprocessor does not inherit from `SwitchablePreprocessor`.
- `compute_ranking_consistency` works with any trained `ScoringModelBase`.
