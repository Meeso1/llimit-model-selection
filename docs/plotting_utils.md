# Plotting (`src/plotting/`)

Plotting utilities for visualising model training histories are organised into a
package with one module of generic helpers and one module per model type.

---

## Module overview

| Module | Model |
|---|---|
| `src/plotting/core.py` | Generic, model-agnostic helpers |
| `src/plotting/response_predictive.py` | `ResponsePredictiveModel` |
| `src/plotting/simple_scoring.py` | `SimpleScoringModel` |
| `src/plotting/elo.py` | `EloScoringModel` |
| `src/plotting/transformer_embedding.py` | `TransformerEmbeddingModel` |
| `src/plotting/dense_network.py` | `DenseNetworkModel` |
| `src/plotting/dn_embedding.py` | `DnEmbeddingModel` |
| `src/plotting/gradient_boosting.py` | `GradientBoostingModel` |
| `src/plotting/dn_length_prediction.py` | `DnEmbeddingLengthPredictionModel` |
| `src/plotting/gb_length_prediction.py` | `GbLengthPredictionModel` |

Non-iterative models (MCMF, least-squares, greedy ranking) have no plotting
module since they produce no epoch-level training history.

---

## `core.py` – generic functions

### Design principles

- **Single-series functions** accept a plain `list[float | None]` and do not
  distinguish between training and validation data — the caller decides which
  series to pass.
- **Combined functions** explicitly accept separate `train_values` and
  `val_values` lists and always plot both.
- `None` entries are silently skipped in all functions.
- Rolling-mean window: `min(10, n // 2)`, adapts to short runs.
- Rolling mean is always drawn in `tab:purple` to avoid clashing with the default blue/orange data lines.

### Single-series functions (with rolling mean + best-value reference line)

| Function | Scale | Best-value line |
|---|---|---|
| `plot_loss(axes, values, title)` | log (normalized) | Min, red dashed |
| `plot_accuracy(axes, values, title)` | −log(1−acc) | Max, red dashed; %-labelled y-axis |
| `plot_positive_metric(axes, values, title, ylabel, show_original_scale)` | log | Min, red dashed |
| `plot_log_variance(axes, values, title)` | log | Min, red dashed |
| `plot_relative_error(axes, values, title)` | log(1+err) | Min, red dashed |
| `plot_ratio_around_one(axes, values, title, ylabel)` | linear | Green ref at 1.0 |
| `plot_delta_metric(axes, values, title, ylabel)` | linear | Green ref at 0 |

**Loss normalization:** `plot_loss` (and the combined loss variants) divide each
series by its first value so the plot always starts at `log(1) = 0`. This is
skipped when the first value is 0.

**Original-scale y-axis:** `plot_positive_metric` and `plot_combined_positive_metric`
accept `show_original_scale=True` to replace log-transformed tick labels with the
actual (pre-log) values — useful for MAE/RMSE displayed in token counts.

### Combined training + validation functions

These overlay both series. Loss and accuracy plots accept `mark_best_val=True`
(the default) to draw a **dashed red reference line** at the best validation
value. Set `mark_best_val=False` for metrics that do not have a clear monotonic
direction (e.g. KL loss, prediction quality).

| Function | Scale | Val reference line |
|---|---|---|
| `plot_combined_loss(axes, train, val, title, mark_best_val)` | log (normalized) | Min val, red dashed |
| `plot_combined_accuracy(axes, train, val, title, mark_best_val)` | −log(1−acc) | Max val, red dashed; %-labelled y-axis |
| `plot_combined_positive_metric(axes, train, val, title, ylabel, show_original_scale)` | log | — |
| `plot_combined_relative_error(axes, train, val, title)` | log(1+err) | — |
| `plot_combined_ratio_around_one(axes, train, val, title, ylabel)` | linear | Green ref at 1.0 |
| `plot_combined_log_variance(axes, train, val, title)` | log | — |

### Multi-series functions

| Function | Description |
|---|---|
| `plot_loss_components(axes, components, title, normalize)` | Multiple loss components on log scale. `normalize=True` divides each series by its first value so improvement trajectories are comparable regardless of absolute magnitude. |
| `plot_accuracy_breakdown(axes, metrics, title)` | Multiple accuracy components on −log(1−acc) scale |
| `plot_distribution_over_time(axes, avg, top, bottom, title, ylabel)` | Fan chart: average + top/bottom 10%; shaded band |

---

## Per-model modules

Each module exposes:

1. **`plot_metrics(log: TrainingLog) -> plt.Figure`** — creates a multi-panel
   figure covering all recorded metrics. No figure-level title is set.
2. **Individual per-metric functions** `plot_<metric>(axes, log)` — thin wrappers
   around `core.py` helpers that pass the correct metric keys and title.

To view a single metric for only one series, call a `core.py` function directly
with the desired values extracted from the log.

### `response_predictive.py`

`plot_metrics` produces a 7 × 2 grid.

| Function | Metric keys | Notes |
|---|---|---|
| `plot_total_loss` | `train_loss` / `val_loss` | |
| `plot_accuracy` | `train_accuracy` / `val_accuracy` | Accuracy uses predicted representations (matches inference) |
| `plot_scorer_real_repr_accuracy` | `scorer_real_repr_accuracy` / val | Upper-bound diagnostic |
| `plot_prediction_quality` | `prediction_quality` / val | No best-val line |
| `plot_repr_mean_variance` | `repr_mean_variance` / val | |
| `plot_predictability_loss` | `predictability_loss` / val | |
| `plot_repr_kl_loss` | `repr_kl_loss` / val | No best-val line |
| `plot_scoring_loss_breakdown` | `real_scoring_loss` / `pred_scoring_loss` (training) | Shows balance between two scoring paths |
| `plot_prediction_loss` | `prediction_loss` (training only) | |
| `plot_pred_scoring_weight` | `pred_scoring_weight` (warmup schedule) | Linear ramp 0→1 over `warmup_epochs` |
| `plot_score_consistency_loss` | `score_consistency_loss` / val | MSE between pred-repr and real-repr scores |
| `plot_repr_dist_kl_loss` | `repr_dist_kl_loss` / val | Symmetric KL between pred/real repr distributions; no best-val line |
| `plot_component_losses_weighted` | scoring, prediction, predictability, repr KL, score consistency, dist KL (training) | Weighted — shows which dominates |
| `plot_component_losses_normalized` | same six (training) | Each normalized to start at 1 — shows improvement rate |

### `simple_scoring.py`

`plot_metrics` produces a 4 × 2 grid.

| Function | Metric keys |
|---|---|
| `plot_loss` | `train_loss` / `val_loss` |
| `plot_accuracy` | `train_accuracy` / `val_accuracy` |
| `plot_train_loss_components` | `ranking_loss`, `tie_loss`, `both_bad_loss` |
| `plot_val_loss_components` | val counterparts |
| `plot_train_accuracy_breakdown` | `ranking_accuracy`, `tie_accuracy`, `both_bad_accuracy` |
| `plot_val_accuracy_breakdown` | val counterparts |
| `plot_score_distribution` | `avg_score`, `top_10_pct_score`, `bottom_10_pct_score` |

### `elo.py`

`plot_metrics` produces a 3 × 2 grid.

| Function | Metric keys |
|---|---|
| `plot_accuracy` | `train_accuracy` / `val_accuracy` |
| `plot_train_accuracy_breakdown` | `ranking_accuracy`, `tie_accuracy`, `both_bad_accuracy` |
| `plot_val_accuracy_breakdown` | val counterparts |
| `plot_rating_distribution` | `avg_rating`, `top_10_pct_rating`, `bottom_10_pct_rating` |
| `plot_rating_change` | `avg_rating_change` |

### `transformer_embedding.py`, `dense_network.py`, `dn_embedding.py`, `gradient_boosting.py`

`plot_metrics` produces a 1 × 2 grid.

| Function | Metric keys |
|---|---|
| `plot_loss` | `train_loss` / `val_loss` |
| `plot_accuracy` | `train_accuracy` / `val_accuracy` |

### `dn_length_prediction.py`, `gb_length_prediction.py`

`plot_metrics` produces a 4 × 2 grid.

| Function | Metric keys | Notes |
|---|---|---|
| `plot_loss` | `train_loss` / `val_loss` | |
| `plot_accuracy` | `train_accuracy` / `val_accuracy` | |
| `plot_mae` | `train_mae` / `val_mae` | Y-axis shows token counts (original scale) |
| `plot_rmse` | `train_rmse` / `val_rmse` | Y-axis shows original scale values |
| `plot_relative_error` | `train_avg_relative_error` / val | |
| `plot_relative_ratio` | `train_avg_relative_ratio` / val | |
| `plot_stddev_ratio` | `train_stddev_ratio` / val | |

---

## Typical usage

```python
from src.plotting import response_predictive
from src.utils.training_logger import load_training_log

log = load_training_log("my-run-name")
fig = response_predictive.plot_metrics(log)
plt.show()
```

Inspect a single metric using only one series:

```python
import matplotlib.pyplot as plt
from src.plotting import core, response_predictive
from src.plotting.core import _get_metric
from src.utils.training_logger import load_training_log

log = load_training_log("my-run-name")
fig, ax = plt.subplots()
core.plot_accuracy(ax, _get_metric(log, 'val_accuracy'), 'Validation Accuracy')
plt.show()
```

Use the generic helpers directly:

```python
from src.plotting.core import plot_combined_loss, plot_accuracy_breakdown

plot_combined_loss(ax, train_vals, val_vals, "My loss")
plot_accuracy_breakdown(ax, {"ranking": ..., "tie": ...}, "Breakdown")
```
