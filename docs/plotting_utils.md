# Plotting (`src/plotting/`)

Plotting utilities for visualising model training histories are organised into a
package with one module of generic helpers and one module per model type.

---

## Module overview

| Module | Model |
|---|---|
| `src/plotting/core.py` | Generic, model-agnostic helpers |
| `src/plotting/triplet_embedding_model.py` | `TripletFrozenEncoderModel` / `TripletFinetunableEncoderModel` embedding logs |
| `src/plotting/attention_embedding_model.py` | `AttentionEmbeddingModel` embedding logs |
| `src/plotting/response_predictive.py` | `ResponsePredictiveModel` |
| `src/plotting/simple_scoring.py` | `SimpleScoringModel` |
| `src/plotting/elo.py` | `EloScoringModel` |
| `src/plotting/transformer_embedding.py` | `TransformerEmbeddingModel` |
| `src/plotting/dense_network.py` | `DenseNetworkModel` |
| `src/plotting/dn_embedding.py` | `DnEmbeddingModel` |
| `src/plotting/gradient_boosting.py` | `GradientBoostingModel` |
| `src/plotting/_gradient_boosting_shared.py` | Shared GB helpers (internal — not for direct import) |
| `src/plotting/_length_prediction_shared.py` | Shared LP helpers (internal — not for direct import) |
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
| `plot_linear_components(axes, components, title, ylabel)` | Multiple series on a **linear** scale (no log). For bounded metrics (e.g. ratios in [0, 1]). |
| `plot_score_variance_decomposition(axes, total_variance, model_ratio, prompt_ratio, title)` | `log(total_variance)` on the left axis and model/prompt variance ratios on a twin axis when both are present. |
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

`plot_metrics` produces a 10 × 2 grid. Per-epoch diagnostics from
`EpochDiagnosticsAccumulator` are included; post-training metrics (e.g.
`sensitivity/*` in `final_metrics`) are not.

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
| `plot_diagnostic_grad_norms` | `encoder_grad_norm`, `predictor_grad_norm`, `scorer_grad_norm` (training) | Normalized log scale |
| `plot_predictor_input_proj_norms` | `predictor_prompt_proj_norm`, `predictor_feat_proj_norm`, `predictor_model_proj_norm` | |
| `plot_pred_real_repr_diagnostics` | `pred_repr_norm`, `real_repr_norm`, `pred_repr_variance`, `real_repr_variance` | |
| `plot_score_variance_diagnostics` | `score_total_variance`, `score_model_variance_ratio`, `score_prompt_variance_ratio` | Real-repr scores; twin-axis |
| `plot_grad_attr_embeddings` | `grad_attr_prompt_embedding`, `grad_attr_model_embedding` | |

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

### `transformer_embedding.py`

`plot_metrics` produces a 4 × 2 grid. Per-epoch diagnostics from
`EpochDiagnosticsAccumulator` are included; post-training metrics (e.g.
`sensitivity/*` in `final_metrics`) are not.

| Function | Metric keys | Notes |
|---|---|---|
| `plot_loss` | `train_loss` / `val_loss` | |
| `plot_accuracy` | `train_accuracy` / `val_accuracy` | |
| `plot_modality_norms` | `prompt_emb_proj_norm`, `feat_proj_norm`, `model_proj_norm` | |
| `plot_gradient_norms` | `transformer_grad_norm`, `projection_grad_norm`, `scoring_head_grad_norm` | Normalized |
| `plot_interaction_norm` | `interaction_norm` | |
| `plot_score_variance` | `score_total_variance`, `score_model_variance_ratio`, `score_prompt_variance_ratio` | Twin-axis |
| `plot_modality_variances` | `prompt_emb_proj_variance`, `feat_proj_variance`, `model_proj_variance`, `interaction_variance` | |
| `plot_grad_attr_embeddings` | `grad_attr_prompt_embedding`, `grad_attr_model_embedding` | |

### `dn_embedding.py`

`plot_metrics` produces a 4 × 2 grid (one panel hidden). Per-epoch diagnostics
from `EpochDiagnosticsAccumulator` are included; post-training metrics (e.g.
`sensitivity/*` in `final_metrics`) are not.

| Function | Metric keys | Notes |
|---|---|---|
| `plot_loss` | `train_loss` / `val_loss` | |
| `plot_accuracy` | `train_accuracy` / `val_accuracy` | |
| `plot_modality_norms` | `prompt_emb_proj_norm`, `prompt_feat_proj_norm`, `model_emb_proj_norm` | |
| `plot_modality_variances` | `*_variance` for the three input projections | |
| `plot_gradient_norms` | `trunk_grad_norm`, `prompt_emb_proj_grad_norm`, `prompt_feat_proj_grad_norm`, `model_emb_proj_grad_norm` | Normalized |
| `plot_score_variance` | `score_total_variance`, `score_model_variance_ratio`, `score_prompt_variance_ratio` | Twin-axis |
| `plot_grad_attr_embeddings` | `grad_attr_prompt_embedding`, `grad_attr_model_embedding` | |

### `dense_network.py`

`plot_metrics` produces a 1 × 2 grid.

| Function | Metric keys |
|---|---|
| `plot_loss` | `train_loss` / `val_loss` |
| `plot_accuracy` | `train_accuracy` / `val_accuracy` |

### `_gradient_boosting_shared.py`

Internal module shared between `gradient_boosting.py` and `gb_length_prediction.py`.
Not intended for direct import. Provides:

| Function | Description |
|---|---|
| `plot_block_importance(axes, log)` | Fraction of total XGBoost gain per input feature block (linear scale). Auto-detects available `importance_<block>` keys. |
| `plot_convergence_diagnostics(axes, log)` | `train_prediction_std` and `tree_contribution_mean` on log scale. |

### `gradient_boosting.py`

`plot_metrics` produces a 2 × 2 grid. After training, the model logs
post-training **`sensitivity/*`** accuracy drops in `final_metrics` (not per-epoch);
those are omitted from `plot_metrics` by design.

| Function | Metric keys | Notes |
|---|---|---|
| `plot_loss` | `train_loss` / `val_loss` | |
| `plot_accuracy` | `train_accuracy` / `val_accuracy` | |
| `plot_block_importance` (via shared) | `importance_<block>` keys | Fraction of total gain per input block |
| `plot_convergence_diagnostics` (via shared) | `train_prediction_std`, `tree_contribution_mean` | Log scale; tree contribution absent at round 1 |

### `dn_length_prediction.py`

`plot_metrics` produces a 5 × 2 grid (10 panels). Per-epoch diagnostics from
`EpochDiagnosticsAccumulator` are included. `plot_grad_attr_embeddings` is
available as a standalone function but not included in `plot_metrics`.
The 7 regression metrics are shared with `gb_length_prediction.py` via
`_length_prediction_shared.py`; `plot_modality_norms`, `plot_modality_variances`,
and `plot_grad_attr_embeddings` are re-exported from `dn_embedding.py`.

| Function | Metric keys | Notes |
|---|---|---|
| `plot_loss` | `train_loss` / `val_loss` | |
| `plot_accuracy` | `train_accuracy` / `val_accuracy` | |
| `plot_mae` | `train_mae` / `val_mae` | Y-axis shows token counts (original scale) |
| `plot_rmse` | `train_rmse` / `val_rmse` | Y-axis shows original scale values |
| `plot_relative_error` | `train_avg_relative_error` / val | |
| `plot_relative_ratio` | `train_avg_relative_ratio` / val | |
| `plot_stddev_ratio` | `train_stddev_ratio` / val | |
| `plot_modality_norms` | `prompt_emb_proj_norm`, `prompt_feat_proj_norm`, `model_emb_proj_norm` | |
| `plot_modality_variances` | `*_variance` for the three input projections | |
| `plot_gradient_norms` | `trunk_grad_norm`, `prompt_emb_proj_grad_norm`, `prompt_feat_proj_grad_norm`, `model_emb_proj_grad_norm`, `model_id_embedding_grad_norm` | Normalized; includes model-id embedding group |
| `plot_grad_attr_embeddings` | `grad_attr_prompt_embedding`, `grad_attr_model_embedding` | Standalone only |

### `triplet_embedding_model.py`

Plots metrics stored in ``TrainingLog.embedding_model_log`` for
``TripletFrozenEncoderModel`` and ``TripletFinetunableEncoderModel``.
**Triplet accuracy** and **nearest-neighbour accuracy** are intentionally
omitted — only loss, universal accuracy, and loss components are shown.

`plot_metrics` produces a 1 × 3 grid.

| Function | Data keys | Notes |
|---|---|---|
| `plot_total_loss` | `train_loss` / `val_loss` | |
| `plot_universal_accuracy` | `train_universal_accuracy` / `val_universal_accuracy` | |
| `plot_loss_components` | `train_triplet_loss`, `train_reg_loss` | Normalized log-scale components |

### `attention_embedding_model.py`

Plots metrics stored in ``TrainingLog.embedding_model_log`` for
``AttentionEmbeddingModel``. **Triplet accuracy** and **nearest-neighbour
accuracy** are intentionally omitted. The contrastive loss has no logged
component breakdown.

`plot_metrics` produces a 1 × 2 grid.

| Function | Data keys |
|---|---|
| `plot_total_loss` | `train_loss` / `val_loss` |
| `plot_universal_accuracy` | `train_universal_accuracy` / `val_universal_accuracy` |

Both modules read from `log.embedding_model_log` using the shared
`_get_embedding_metric` helper in `core.py`.

### `gb_length_prediction.py`

`plot_metrics` produces a 5 × 2 grid (last panel hidden). Shares the 7
regression metric functions with `dn_length_prediction.py` via
`_length_prediction_shared.py`. Diagnostic panels come from `_gradient_boosting_shared.py`.

| Function | Metric keys | Notes |
|---|---|---|
| `plot_loss` | `train_loss` / `val_loss` | |
| `plot_accuracy` | `train_accuracy` / `val_accuracy` | |
| `plot_mae` | `train_mae` / `val_mae` | Y-axis shows token counts (original scale) |
| `plot_rmse` | `train_rmse` / `val_rmse` | Y-axis shows original scale values |
| `plot_relative_error` | `train_avg_relative_error` / val | |
| `plot_relative_ratio` | `train_avg_relative_ratio` / val | |
| `plot_stddev_ratio` | `train_stddev_ratio` / val | |
| `plot_convergence_diagnostics` (via shared) | `train_prediction_std`, `tree_contribution_mean` | Log scale |
| `plot_block_importance` (via shared) | `importance_<block>` keys | Fraction of total gain per input block |

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
