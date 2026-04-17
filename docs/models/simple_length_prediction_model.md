# Simple Length Prediction Model

A lightweight, non-iterative benchmark model for response length prediction. Uses per-model ordinary least squares (OLS) linear regression with features selected by name from the existing 45 prompt features. No model embeddings are required. Intended as a baseline to compare more sophisticated models against.

---

## Concept

For each model seen during training, fit:

```
y = X @ w + b
```

where `y` is the standardised log-response-length and `X` contains a configurable subset of the pre-computed prompt features. The feature set is controlled by `input_features` -- a list of feature names from `get_feature_descriptions()`.

- **Empty list** (`[]`, default) -- intercept-only: predicts the per-model mean response length, ignoring the prompt entirely.
- **`["Word count"]`** -- per-model linear regression from prompt word count.
- **`["Word count", "Sentence count", "Has code request"]`** -- multivariate regression mixing numeric and boolean features.

For models not seen during training, predictions fall back to a global regression fitted on all training data pooled.

---

## Feature selection

Features are referenced by the `.name` attribute of the descriptors returned by `get_feature_descriptions()` in `src/preprocessing/scoring_feature_extraction.py`. All 45 features are available:

**Numeric (27):** Character count, Word count, Sentence count, Inverse Type-token ratio, Average word length, Question density, Nesting depth, Multi-part request density, 12 domain scores (Science, Medicine, Law, Finance, Tech, Academic, Casual, Formal, Philosophical, Historical, Personal, Business), Informal marker density, Specificity density, Politeness density, Urgency density, Conversation turn count, Total context length, Assistant turn count.

**Boolean (18):** Has code request, Has code block, Has math, Has numbers, Has math symbols, Has creative writing, Has factual query, Has instruction following, Has roleplay, Has analysis, Starts with verb, Is question, Is follow-up, Expects list, Expects table, Expects JSON, Expects code, Expects long response.

The `use_scaled_features` parameter controls whether the transformed (log/softplus + standardised) features from the preprocessor are used as-is (`True`, default) or fully inverted back toward raw space first (`False`, via `inverse_transform_prompt_features()`). Both options work for OLS; `True` is the default since the features are already available in that form.

---

## Preprocessing

Delegates entirely to `LengthPredictionPreprocessor`:

- Filters rare models (`min_model_comparisons`), empty entries, and outlier responses.
- Computes response lengths via `word_count * 1.3` token heuristic, then `log` + `SimpleScaler` standardisation.
- Produces `prompt_features [45]` per sample (already transformed/scaled).
- Caches preprocessed data in `Jars.preprocessed_data` -- if any other length prediction model has already been trained on the same dataset, no recomputation is needed.

Note: the preprocessor also computes sentence-transformer embeddings (needed by other models) which are ignored here. This is free if the data is already cached.

---

## Training

Non-iterative (single pass). The `epochs` parameter is ignored.

1. Run `LengthPredictionPreprocessor.preprocess(data)`.
2. Resolve `input_features` names to column indices in the `[45]` feature vector.
3. Optionally unscale features (`use_scaled_features=False`).
4. Split into train/val via `ValidationSplit`.
5. For each model, fit OLS on (selected features, scaled log-length) pairs.
6. Fit a global fallback OLS on all pooled training data.
7. Compute train/val metrics and log a single `TrainingHistoryEntry`.

---

## Prediction

For each `(prompt, model)` pair:

1. Call `preprocessor.preprocess_for_inference()` to get `prompt_features`.
2. Select configured feature columns.
3. Apply per-model `(w, b)`; if model is unseen, use the global fallback.
4. Inverse-scale + `exp` to convert from standardised log-space to raw token count.

---

## Metrics

Same as other length prediction models (see `compute_length_prediction_metrics`):

| Metric | Description |
|---|---|
| `rmse` | Root mean squared error in standardised log-space |
| `mae` | Mean absolute error in raw token-count space |
| `avg_relative_error` | Mean \|1 − predicted/actual\| |
| `avg_relative_ratio` | Mean predicted/actual (1.0 = perfect) |
| `stddev_ratio` | stddev(predictions) / stddev(actuals) (1.0 = perfect) |
| `accuracy` | Synthetic: `1 / 2^avg_relative_error` |

---

## State dict

| Key | Type | Description |
|---|---|---|
| `input_features` | `list[str]` | Feature names configured at construction |
| `use_scaled_features` | `bool` | Whether scaled features are used |
| `embedding_model_name` | `str` | ST model name (for preprocessor) |
| `min_model_comparisons` | `int` | Filtering threshold |
| `print_every` | `int \| None` | Print frequency |
| `feature_indices` | `list[int]` | Resolved column indices |
| `output_scaler_state` | `dict` | `SimpleScaler` for log-lengths |
| `prompt_features_scaler_state` | `dict` | `SimpleScaler` for prompt features |
| `model_encoder_state` | `dict` | `StringEncoder` fitted during training |
| `per_model_params` | `dict[str, {"weights": ndarray, "bias": float}]` | Per-model OLS coefficients |
| `global_params` | `{"weights": ndarray, "bias": float}` | Fallback OLS coefficients |

---

## Specification

```python
class SimpleLengthPredictionSpecification(ModelSpecBase):
    model_type: Literal["simple_length_prediction"] = "simple_length_prediction"
    input_features: list[str] = []
    use_scaled_features: bool = True
    embedding_model_name: str = "all-MiniLM-L6-v2"
    min_model_comparisons: int = 20
```

---

## CLI example

Per-model mean (no features):

```json
{
  "model": {
    "name": "simple_lp_mean",
    "spec": {
      "model_type": "simple_length_prediction",
      "input_features": [],
      "min_model_comparisons": 20
    }
  },
  "data": {
    "dataset": "lmarena_human_preference",
    "max_samples": null,
    "validation_split": 0.1,
    "seed": 42
  },
  "log": { "run_name": null, "print_every": 1 },
  "epochs": 1,
  "batch_size": 32
}
```

Per-model regression from word count:

```json
{
  "model": {
    "name": "simple_lp_word_count",
    "spec": {
      "model_type": "simple_length_prediction",
      "input_features": ["Word count"],
      "use_scaled_features": true,
      "min_model_comparisons": 20
    }
  },
  "data": {
    "dataset": "lmarena_human_preference",
    "max_samples": null,
    "validation_split": 0.1,
    "seed": 42
  },
  "log": { "run_name": null, "print_every": 1 },
  "epochs": 1,
  "batch_size": 32
}
```

```bash
python -m src.scripts.cli train --spec-file spec.json
```
