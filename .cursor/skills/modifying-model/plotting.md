# Plotting Utilities Changes

Plotting for each model lives in `src/plotting/<model_name>.py`. The docs live in `docs/plotting_utils.md`.

## File to edit

`src/plotting/response_predictive.py` (or the equivalent for the model being changed).

## Structure

Each plotting module has:
1. **`plot_metrics(log)`** — creates the full multi-panel figure. Update the grid size and the cell assignments when adding/removing panels.
2. **Individual `plot_<metric>()` functions** — one per panel. Add new ones, remove old ones.
3. **`plot_component_losses_weighted` / `plot_component_losses_normalized`** — if the model has component loss tracking, update the dict of components to include new loss terms (with their weights read from `log.config`).

## Adding a new metric panel

1. Write a `plot_<metric>` function using the appropriate `core.py` helper (see below).
2. Add it to `plot_metrics` and update the grid dimensions.

## Removing a metric panel

Delete the function and remove it from `plot_metrics`. Update the grid dimensions.

## Choosing the right core helper

| Metric type | Core helper | `mark_best_val` |
|---|---|---|
| Loss (lower is better) | `_plot_combined_loss` | `True` (default) |
| Loss with no clear direction (KL, diversity) | `_plot_combined_loss` | `False` |
| Accuracy / quality in [0,1] | `_plot_combined_accuracy` | `True` or `False` |
| Variance / positive metric | `_plot_combined_log_variance` | — |
| Training-only single series | `_plot_loss` | — |
| Multiple components on one axes | `_plot_loss_components(normalize=False)` | — |
| Normalized improvement trajectories | `_plot_loss_components(normalize=True)` | — |
| Simple line (schedule, ratio) | `_filter_nones` + `axes.plot(...)` manually | — |

## Reading weights from config

For component loss plots, read weights from `log.config` with a fallback to the constructor default:

```python
scw = float(log.config.get('score_consistency_loss_weight', 0.1))
```

## Updating docs

After changing `src/plotting/response_predictive.py`, update the matching table in `docs/plotting_utils.md` under `### response_predictive.py`. Update:
- The grid size (`plot_metrics produces a N × 2 grid`)
- The function/metric/notes table rows

The table columns are: `Function | Metric keys | Notes`.
