# Plotting Utilities (`src/plotting_utils.py`)

Helper functions for visualising model training histories.  All functions
that plot a single series apply a **rolling mean** to smooth the signal; all
functions that overlay two or more series (e.g. train + val) omit the rolling
mean so the two curves remain distinguishable.

The rolling-window size is `min(10, n // 2)`, so it adapts gracefully to short
runs.

---

## Generic single-trace functions

These accept a single `list[float | None]` where `None` entries (epochs with no
value) are skipped automatically.

| Function | Scale | Notes |
|---|---|---|
| `plot_loss(axes, values, title)` | log | Min reference line |
| `plot_accuracy(axes, values, title)` | −log(1−acc) | %-labelled y-axis, max reference line |
| `plot_positive_metric(axes, values, title, ylabel)` | log | For MAE, RMSE, or any positive-valued metric |
| `plot_relative_error(axes, values, title)` | log(1+err) | Lower-is-better; min reference line |
| `plot_ratio_around_one(axes, values, title, ylabel)` | linear | Reference line at 1.0; for `stddev_ratio`, `avg_relative_ratio` |
| `plot_delta_metric(axes, values, title, ylabel)` | linear | Reference line at 0; for `avg_rating_change` or similar |

---

## Generic train + validation combined functions

These accept two lists (`train_values`, `val_values`) and plot both on the
same axes.  No rolling mean is applied.

| Function | Scale | Notes |
|---|---|---|
| `plot_train_val_loss(axes, train, val, title)` | log | — |
| `plot_train_val_accuracy(axes, train, val, title)` | −log(1−acc) | %-labelled y-axis |
| `plot_train_val_positive_metric(axes, train, val, title, ylabel)` | log | For MAE, RMSE, etc. |
| `plot_train_val_relative_error(axes, train, val, title)` | log(1+err) | — |
| `plot_train_val_ratio_around_one(axes, train, val, title, ylabel)` | linear | Reference line at 1.0 |
| `plot_train_val_log_variance(axes, train, val, title)` | log | For `repr_mean_variance` |

---

## Multi-series functions

These accept a `dict[str, list[float | None]]` (or three separate lists) and
overlay all series on one axes.  No rolling mean.

| Function | Description |
|---|---|
| `plot_loss_components(axes, components, title)` | Multiple loss components on log scale |
| `plot_accuracy_breakdown(axes, metrics, title)` | Multiple accuracy components on −log(1−acc) scale |
| `plot_distribution_over_time(axes, avg, top, bottom, title, ylabel)` | Fan chart: average + top/bottom 10 %; shaded band between extremes |

---

## Model-specific figure functions

These create and return a `plt.Figure` covering all relevant metrics for one
model type.  Pass a `TrainingLog` loaded with `load_training_log(run_name)` from
`src.utils.training_logger`.

### `plot_length_prediction_history(log) → Figure`

4 × 2 grid (last panel hidden):

| Panel | Content |
|---|---|
| [0,0] | Train/val loss |
| [0,1] | Train/val accuracy |
| [1,0] | Train/val MAE (log) |
| [1,1] | Train/val RMSE (log) |
| [2,0] | Train/val relative error |
| [2,1] | Train/val relative ratio (ref 1.0) |
| [3,0] | Train/val stddev ratio (ref 1.0) |

Metrics read from `additional_metrics`: `train_mae`, `val_mae`, `train_rmse`,
`val_rmse`, `train_avg_relative_error`, `val_avg_relative_error`,
`train_avg_relative_ratio`, `val_avg_relative_ratio`, `train_stddev_ratio`,
`val_stddev_ratio`.

---

### `plot_response_predictive_history(log) → Figure`

5 × 2 grid:

| Panel | Content |
|---|---|
| [0,0] | Train/val total loss |
| [0,1] | Train/val accuracy |
| [1,0] | Train/val scorer real repr accuracy |
| [1,1] | Train/val prediction quality |
| [2,0] | Train/val repr mean variance (log) |
| [2,1] | Train/val predictability loss |
| [3,0] | Train/val repr KL loss |
| [3,1] | Scoring loss (train, rolling mean) |
| [4,0] | Prediction loss (train, rolling mean) |
| [4,1] | Current real representation ratio (curriculum schedule) |

---

### `plot_simple_scoring_history(log) → Figure`

4 × 2 grid (last panel hidden):

| Panel | Content |
|---|---|
| [0,0] | Train/val loss |
| [0,1] | Train/val accuracy |
| [1,0] | Train loss components (ranking / tie / both_bad) |
| [1,1] | Val loss components |
| [2,0] | Train accuracy breakdown |
| [2,1] | Val accuracy breakdown |
| [3,0] | Score distribution: avg, top 10%, bottom 10% |

---

### `plot_elo_history(log) → Figure`

3 × 2 grid (last panel hidden):

| Panel | Content |
|---|---|
| [0,0] | Train/val accuracy |
| [0,1] | Train accuracy breakdown (ranking / tie / both_bad) |
| [1,0] | Val accuracy breakdown |
| [1,1] | ELO rating distribution: avg, top 10%, bottom 10% |
| [2,0] | Average rating change per epoch (Δ rating, ref 0) |

---

## Typical usage

```python
from src.plotting_utils import plot_response_predictive_history
from src.utils.training_logger import load_training_log

log = load_training_log("my-run-name")
fig = plot_response_predictive_history(log)
plt.show()
```

For more targeted analysis you can also call the lower-level functions directly:

```python
from src.plotting_utils import plot_train_val_loss, plot_accuracy_breakdown

plot_train_val_loss(ax, history.total_loss, history.val_loss, "My loss")
plot_accuracy_breakdown(ax, {
    "ranking": history.additional_metrics["ranking_accuracy"],
    "tie":     history.additional_metrics["tie_accuracy"],
}, "My breakdown")
```
