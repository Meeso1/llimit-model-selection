import matplotlib.pyplot as plt
import numpy as np

from src.utils.training_logger import TrainingLog


_ROLLING_WINDOW = 10
_ROLLING_MEAN_COLOR = 'tab:purple'


# ---------------------------------------------------------------------------
# Single-series functions (rolling mean + best-value reference line)
# ---------------------------------------------------------------------------

def plot_loss(axes: plt.Axes, values: list[float | None], title: str) -> None:
    """Plot a single loss series.

    Loss is normalized so the first value is 1 (unless the first value is 0).
    Plotted on log scale with rolling mean and a dashed red line at the minimum.
    """
    idx, vals = _filter_nones(values)
    if len(vals) == 0:
        axes.set_title(title)
        return

    vals = _normalize_loss(vals)
    log_loss = np.log(np.maximum(vals, 1e-10))
    window = _rolling_window(len(vals))

    axes.plot(idx, log_loss, alpha=0.6)

    if window >= 2:
        sliding_mean = np.convolve(log_loss, np.ones(window) / window, mode='valid')
        axes.plot(idx[window - 1:], sliding_mean, color=_ROLLING_MEAN_COLOR, label=f"Rolling mean ({window})")
        min_val = np.min(sliding_mean)
        axes.axhline(y=min_val, color='r', linestyle='--', label=f'Best: {np.exp(min_val):.4g}')

    axes.set_ylabel("log(loss)")
    axes.set_title(title)
    axes.legend()


def plot_accuracy(axes: plt.Axes, values: list[float | None], title: str) -> None:
    """Plot a single accuracy series.

    Plotted on -log(1-acc) scale with rolling mean and a dashed red line at the maximum.
    """
    idx, vals = _filter_nones(values)
    if len(vals) == 0:
        axes.set_title(title)
        return

    inv_log_acc = -np.log(1 - vals + 1e-10)
    window = _rolling_window(len(vals))

    axes.plot(idx, inv_log_acc, alpha=0.6)

    if window >= 2:
        sliding_mean = np.convolve(inv_log_acc, np.ones(window) / window, mode='valid')
        axes.plot(idx[window - 1:], sliding_mean, color=_ROLLING_MEAN_COLOR, label=f"Rolling mean ({window})")
        max_val = np.max(sliding_mean)
        max_accuracy = 1 - np.exp(-max_val)
        axes.axhline(y=max_val, color='r', linestyle='--', label=f'Best: {max_accuracy * 100:.1f}%')

    _set_accuracy_yticks(axes, vals)
    axes.set_ylabel("Accuracy")
    axes.set_title(title)
    axes.legend()


def plot_positive_metric(
    axes: plt.Axes,
    values: list[float | None],
    title: str,
    ylabel: str = "log(value)",
    show_original_scale: bool = False,
    skip_first_n_epochs: int = 0,
) -> None:
    """Plot a single positive metric with log scale and rolling mean.

    Suitable for MAE, RMSE, or any positive-valued metric that decreases over training.
    When show_original_scale is True, y-axis ticks show the original (pre-log) values.
    When skip_first_n_epochs > 0, the y-axis range is computed from epoch
    skip_first_n_epochs onward (all epochs are still plotted).
    """
    idx, vals = _filter_nones(values)
    if len(vals) == 0:
        axes.set_title(title)
        return

    log_vals = np.log(np.maximum(vals, 1e-10))
    window = _rolling_window(len(vals))

    axes.plot(idx, log_vals, alpha=0.6)

    if window >= 2:
        rolling = np.convolve(log_vals, np.ones(window) / window, mode='valid')
        axes.plot(idx[window - 1:], rolling, color=_ROLLING_MEAN_COLOR, label=f"Rolling mean ({window})")
        min_val = np.min(rolling)
        axes.axhline(y=min_val, color='r', linestyle='--', label=f'Best: {np.exp(min_val):.4g}')

    y_range = _ylim_for_tail([(idx, log_vals)], skip_first_n_epochs) if skip_first_n_epochs > 0 else None
    if show_original_scale:
        _set_log_yticks_with_original_scale(axes, log_vals, log_range=y_range)
    if y_range is not None:
        axes.set_ylim(*y_range)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()


def plot_log_variance(axes: plt.Axes, values: list[float | None], title: str) -> None:
    """Plot a single variance metric with log scale and rolling mean."""
    plot_positive_metric(axes, values, title, ylabel="log(variance)")


def plot_relative_error(
    axes: plt.Axes,
    values: list[float | None],
    title: str,
    skip_first_n_epochs: int = 0,
) -> None:
    """Plot relative error with log(1+error) scale and rolling mean.

    Lower is better. The log(1+error) transform expands the scale near 0.
    When skip_first_n_epochs > 0, the y-axis range is computed from epoch
    skip_first_n_epochs onward (all epochs are still plotted).
    """
    idx, vals = _filter_nones(values)
    if len(vals) == 0:
        axes.set_title(title)
        return

    log_err = np.log1p(vals)
    window = _rolling_window(len(vals))

    axes.plot(idx, log_err, alpha=0.6)

    if window >= 2:
        rolling = np.convolve(log_err, np.ones(window) / window, mode='valid')
        axes.plot(idx[window - 1:], rolling, color=_ROLLING_MEAN_COLOR, label=f"Rolling mean ({window})")
        min_rolling = np.min(rolling)
        axes.axhline(y=min_rolling, color='r', linestyle='--', label=f'Best: {np.expm1(min_rolling):.3f}')

    if skip_first_n_epochs > 0:
        y_range = _ylim_for_tail([(idx, log_err)], skip_first_n_epochs)
        if y_range is not None:
            axes.set_ylim(*y_range)
    axes.set_ylabel("log(1 + relative error)")
    axes.set_title(title)
    axes.legend()


def plot_ratio_around_one(
    axes: plt.Axes,
    values: list[float | None],
    title: str,
    ylabel: str = "Ratio",
    skip_first_n_epochs: int = 0,
) -> None:
    """Plot a ratio metric that should ideally be 1.0 with rolling mean.

    Shows a green reference line at 1.0. Suitable for stddev_ratio, avg_relative_ratio.
    When skip_first_n_epochs > 0, the y-axis range is computed from epoch
    skip_first_n_epochs onward (all epochs are still plotted).
    """
    idx, vals = _filter_nones(values)
    if len(vals) == 0:
        axes.set_title(title)
        return

    window = _rolling_window(len(vals))

    axes.plot(idx, vals, alpha=0.6)

    if window >= 2:
        rolling = np.convolve(vals, np.ones(window) / window, mode='valid')
        axes.plot(idx[window - 1:], rolling, color=_ROLLING_MEAN_COLOR, label=f"Rolling mean ({window})")

    if skip_first_n_epochs > 0:
        y_range = _ylim_for_tail([(idx, vals)], skip_first_n_epochs)
        if y_range is not None:
            axes.set_ylim(*y_range)
    axes.axhline(y=1.0, color='g', linestyle='--', label='Ideal (1.0)', alpha=0.7)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()


def plot_delta_metric(
    axes: plt.Axes,
    values: list[float | None],
    title: str,
    ylabel: str = "Value",
) -> None:
    """Plot a metric that can be positive or negative (e.g. rating change) with rolling mean.

    Shows a green reference line at 0.
    """
    idx, vals = _filter_nones(values)
    if len(vals) == 0:
        axes.set_title(title)
        return

    window = _rolling_window(len(vals))

    axes.plot(idx, vals, alpha=0.6)

    if window >= 2:
        rolling = np.convolve(vals, np.ones(window) / window, mode='valid')
        axes.plot(idx[window - 1:], rolling, color=_ROLLING_MEAN_COLOR, label=f"Rolling mean ({window})")

    axes.axhline(y=0.0, color='g', linestyle='--', label='Zero', alpha=0.7)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()


# ---------------------------------------------------------------------------
# Combined training + validation functions
# ---------------------------------------------------------------------------

def plot_combined_loss(
    axes: plt.Axes,
    train_values: list[float | None],
    val_values: list[float | None],
    title: str,
    mark_best_val: bool = True,
) -> None:
    """Plot training and validation loss on log scale.

    Each series is independently normalized so its first value is 1 (unless the
    first value is 0). When mark_best_val is True, a dashed red reference line
    marks the minimum validation value.
    """
    train_idx, train_vals = _filter_nones(train_values)
    val_idx, val_vals = _filter_nones(val_values)

    if len(train_vals) == 0 and len(val_vals) == 0:
        axes.set_title(title)
        return

    if len(train_vals) > 0:
        train_norm = _normalize_loss(train_vals)
        axes.plot(train_idx, np.log(np.maximum(train_norm, 1e-10)), label="Training", alpha=0.8)

    if len(val_vals) > 0:
        val_norm = _normalize_loss(val_vals)
        log_val = np.log(np.maximum(val_norm, 1e-10))
        axes.plot(val_idx, log_val, label="Validation", alpha=0.8)
        if mark_best_val:
            min_val = np.min(log_val)
            axes.axhline(y=min_val, color='r', linestyle='--', label=f'Best val: {np.exp(min_val):.4g}')

    axes.set_ylabel("log(loss)")
    axes.set_title(title)
    axes.legend()


def plot_combined_accuracy(
    axes: plt.Axes,
    train_values: list[float | None],
    val_values: list[float | None],
    title: str,
    mark_best_val: bool = True,
) -> None:
    """Plot training and validation accuracy on -log(1-acc) scale.

    When mark_best_val is True, a dashed red reference line marks the maximum
    validation accuracy.
    """
    train_idx, train_vals = _filter_nones(train_values)
    val_idx, val_vals = _filter_nones(val_values)

    if len(train_vals) == 0 and len(val_vals) == 0:
        axes.set_title(title)
        return

    if len(train_vals) > 0:
        axes.plot(train_idx, -np.log(1 - train_vals + 1e-10), label="Training", alpha=0.8)

    if len(val_vals) > 0:
        inv_log_val = -np.log(1 - val_vals + 1e-10)
        axes.plot(val_idx, inv_log_val, label="Validation", alpha=0.8)
        if mark_best_val:
            max_val = np.max(inv_log_val)
            max_acc = 1 - np.exp(-max_val)
            axes.axhline(y=max_val, color='r', linestyle='--', label=f'Best val: {max_acc * 100:.1f}%')

    all_vals = np.concatenate([train_vals, val_vals])
    _set_accuracy_yticks(axes, all_vals)
    axes.set_ylabel("Accuracy")
    axes.set_title(title)
    axes.legend()


def plot_combined_positive_metric(
    axes: plt.Axes,
    train_values: list[float | None],
    val_values: list[float | None],
    title: str,
    ylabel: str = "log(value)",
    show_original_scale: bool = False,
    skip_first_n_epochs: int = 0,
) -> None:
    """Plot training and validation of any positive metric on log scale.

    Suitable for MAE, RMSE, or similar metrics.
    When show_original_scale is True, y-axis ticks show the original (pre-log) values.
    When skip_first_n_epochs > 0, the y-axis range is computed from epoch
    skip_first_n_epochs onward (all epochs are still plotted).
    """
    train_idx, train_vals = _filter_nones(train_values)
    val_idx, val_vals = _filter_nones(val_values)

    if len(train_vals) == 0 and len(val_vals) == 0:
        axes.set_title(title)
        return

    log_train = np.log(np.maximum(train_vals, 1e-10)) if len(train_vals) > 0 else np.array([])
    log_val = np.log(np.maximum(val_vals, 1e-10)) if len(val_vals) > 0 else np.array([])

    if len(log_train) > 0:
        axes.plot(train_idx, log_train, label="Training", alpha=0.8)
    if len(log_val) > 0:
        axes.plot(val_idx, log_val, label="Validation", alpha=0.8)

    all_log_vals = np.concatenate([log_train, log_val]) if len(log_train) + len(log_val) > 0 else np.array([])

    tail_series = []
    if len(log_train) > 0:
        tail_series.append((train_idx, log_train))
    if len(log_val) > 0:
        tail_series.append((val_idx, log_val))
    y_range = _ylim_for_tail(tail_series, skip_first_n_epochs) if skip_first_n_epochs > 0 and tail_series else None

    if show_original_scale and len(all_log_vals) > 0:
        _set_log_yticks_with_original_scale(axes, all_log_vals, log_range=y_range)
    if y_range is not None:
        axes.set_ylim(*y_range)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()


def plot_combined_relative_error(
    axes: plt.Axes,
    train_values: list[float | None],
    val_values: list[float | None],
    title: str,
    skip_first_n_epochs: int = 0,
) -> None:
    """Plot training and validation relative error using log(1+error) scale.

    The log(1+error) transform expands the scale near 0 where improvements matter most.
    When skip_first_n_epochs > 0, the y-axis range is computed from epoch
    skip_first_n_epochs onward (all epochs are still plotted).
    """
    train_idx, train_vals = _filter_nones(train_values)
    val_idx, val_vals = _filter_nones(val_values)

    if len(train_vals) == 0 and len(val_vals) == 0:
        axes.set_title(title)
        return

    log_train = np.log1p(train_vals) if len(train_vals) > 0 else np.array([])
    log_val = np.log1p(val_vals) if len(val_vals) > 0 else np.array([])

    if len(log_train) > 0:
        axes.plot(train_idx, log_train, label="Training", alpha=0.8)
    if len(log_val) > 0:
        axes.plot(val_idx, log_val, label="Validation", alpha=0.8)

    if skip_first_n_epochs > 0:
        tail_series = []
        if len(log_train) > 0:
            tail_series.append((train_idx, log_train))
        if len(log_val) > 0:
            tail_series.append((val_idx, log_val))
        y_range = _ylim_for_tail(tail_series, skip_first_n_epochs) if tail_series else None
        if y_range is not None:
            axes.set_ylim(*y_range)
    axes.set_ylabel("log(1 + relative error)")
    axes.set_title(title)
    axes.legend()


def plot_combined_ratio_around_one(
    axes: plt.Axes,
    train_values: list[float | None],
    val_values: list[float | None],
    title: str,
    ylabel: str = "Ratio",
    skip_first_n_epochs: int = 0,
) -> None:
    """Plot training and validation of a ratio that should ideally be 1.0.

    Displays a green reference line at 1.0.
    Suitable for stddev_ratio, avg_relative_ratio.
    When skip_first_n_epochs > 0, the y-axis range is computed from epoch
    skip_first_n_epochs onward (all epochs are still plotted).
    """
    train_idx, train_vals = _filter_nones(train_values)
    val_idx, val_vals = _filter_nones(val_values)

    if len(train_vals) == 0 and len(val_vals) == 0:
        axes.set_title(title)
        return

    if len(train_vals) > 0:
        axes.plot(train_idx, train_vals, label="Training", alpha=0.8)
    if len(val_vals) > 0:
        axes.plot(val_idx, val_vals, label="Validation", alpha=0.8)

    if skip_first_n_epochs > 0:
        tail_series = []
        if len(train_vals) > 0:
            tail_series.append((train_idx, train_vals))
        if len(val_vals) > 0:
            tail_series.append((val_idx, val_vals))
        y_range = _ylim_for_tail(tail_series, skip_first_n_epochs) if tail_series else None
        if y_range is not None:
            axes.set_ylim(*y_range)
    axes.axhline(y=1.0, color='g', linestyle='--', label='Ideal (1.0)', alpha=0.7)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()


def plot_combined_log_variance(
    axes: plt.Axes,
    train_values: list[float | None],
    val_values: list[float | None],
    title: str,
) -> None:
    """Plot training and validation variance in log scale."""
    train_idx, train_vals = _filter_nones(train_values)
    val_idx, val_vals = _filter_nones(val_values)

    if len(train_vals) == 0 and len(val_vals) == 0:
        axes.set_title(title)
        return

    if len(train_vals) > 0:
        axes.plot(train_idx, np.log(np.maximum(train_vals, 1e-10)), label="Training", alpha=0.8)
    if len(val_vals) > 0:
        axes.plot(val_idx, np.log(np.maximum(val_vals, 1e-10)), label="Validation", alpha=0.8)

    axes.set_ylabel("log(variance)")
    axes.set_title(title)
    axes.legend()


# ---------------------------------------------------------------------------
# Multi-series functions
# ---------------------------------------------------------------------------

def plot_loss_components(
    axes: plt.Axes,
    components: dict[str, list[float | None]],
    title: str,
    normalize: bool = False,
    ylabel: str = "log(loss)",
) -> None:
    """Plot multiple loss components on the same axes in log scale.

    When normalize is True, each series is independently normalized so its first
    value is 1, making relative improvement trajectories comparable across
    components that may have very different absolute magnitudes.
    """
    has_data = False
    for name, values in components.items():
        idx, vals = _filter_nones(values)
        if len(vals) > 0:
            if normalize:
                vals = _normalize_loss(vals)
            axes.plot(idx, np.log(np.maximum(vals, 1e-10)), label=name, alpha=0.8)
            has_data = True

    if not has_data:
        axes.set_title(title)
        return

    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()


def plot_linear_components(
    axes: plt.Axes,
    components: dict[str, list[float | None]],
    title: str,
    ylabel: str = "Value",
) -> None:
    """Plot multiple series on a linear scale (no log transform).

    Use for bounded metrics (e.g. variance ratios in [0, 1]) where log scale is
    inappropriate.  Skips series with no data.
    """
    has_data = False
    for name, values in components.items():
        idx, vals = _filter_nones(values)
        if len(vals) > 0:
            axes.plot(idx, vals, label=name, alpha=0.8)
            has_data = True

    if not has_data:
        axes.set_title(title)
        return

    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()


def plot_score_variance_decomposition(
    axes: plt.Axes,
    total_variance: list[float | None],
    model_ratio: list[float | None],
    prompt_ratio: list[float | None],
    title: str = "Score variance decomposition (Training)",
) -> None:
    """Plot log(total batch score variance) and model/prompt variance-ratio lines.

    Ratios are only defined when a diagnostic batch has ≥2 distinct models; missing
    epochs appear as gaps.  Uses a twin y-axis when both total variance and ratios
    are present.
    """
    idx_t, vals_t = _filter_nones(total_variance)
    idx_m, vals_m = _filter_nones(model_ratio)
    idx_p, vals_p = _filter_nones(prompt_ratio)

    if len(vals_t) == 0 and len(vals_m) == 0 and len(vals_p) == 0:
        axes.set_title(title)
        return

    legend_lines: list = []
    legend_labels: list[str] = []

    if len(vals_t) > 0:
        (ln,) = axes.plot(idx_t, np.log(np.maximum(vals_t, 1e-10)), color='tab:blue', alpha=0.8)
        legend_lines.append(ln)
        legend_labels.append('log(total variance)')
        axes.set_ylabel('log(total variance)')

    if len(vals_m) > 0 or len(vals_p) > 0:
        ax2 = axes.twinx() if len(vals_t) > 0 else axes
        if len(vals_m) > 0:
            (ln,) = ax2.plot(idx_m, vals_m, color='tab:orange', alpha=0.8)
            legend_lines.append(ln)
            legend_labels.append('Model var ratio')
        if len(vals_p) > 0:
            (ln,) = ax2.plot(idx_p, vals_p, color='tab:green', alpha=0.8)
            legend_lines.append(ln)
            legend_labels.append('Prompt var ratio')
        ax2.set_ylabel('Variance ratio')

    if legend_lines:
        axes.legend(legend_lines, legend_labels, loc='best')
    axes.set_title(title)


def plot_accuracy_breakdown(
    axes: plt.Axes,
    metrics: dict[str, list[float | None]],
    title: str,
) -> None:
    """Plot multiple accuracy components together with -log(1-acc) scale.

    Suitable for ranking/tie/both_bad accuracy breakdown.
    """
    series_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_vals: list[float] = []

    for name, values in metrics.items():
        idx, vals = _filter_nones(values)
        if len(vals) > 0:
            series_data[name] = (idx, vals)
            all_vals.extend(vals.tolist())

    if not series_data:
        axes.set_title(title)
        return

    for name, (idx, vals) in series_data.items():
        axes.plot(idx, -np.log(1 - vals + 1e-10), label=name, alpha=0.8)

    _set_accuracy_yticks(axes, np.array(all_vals))
    axes.set_ylabel("Accuracy")
    axes.set_title(title)
    axes.legend()


def plot_distribution_over_time(
    axes: plt.Axes,
    avg_values: list[float | None],
    top_values: list[float | None],
    bottom_values: list[float | None],
    title: str,
    ylabel: str = "Value",
) -> None:
    """Plot distribution of a metric over time: average, top 10%, and bottom 10%.

    Shades the region between top and bottom to visualise the spread.
    """
    avg_idx, avg_vals = _filter_nones(avg_values)
    top_idx, top_vals = _filter_nones(top_values)
    bot_idx, bot_vals = _filter_nones(bottom_values)

    if len(avg_vals) == 0 and len(top_vals) == 0 and len(bot_vals) == 0:
        axes.set_title(title)
        return

    if len(avg_vals) > 0:
        axes.plot(avg_idx, avg_vals, label="Average", linewidth=2)
    if len(top_vals) > 0:
        axes.plot(top_idx, top_vals, label="Top 10%", linestyle='--', alpha=0.7)
    if len(bot_vals) > 0:
        axes.plot(bot_idx, bot_vals, label="Bottom 10%", linestyle='--', alpha=0.7)

    if len(top_vals) > 0 and len(bot_vals) > 0 and np.array_equal(top_idx, bot_idx):
        axes.fill_between(top_idx, bot_vals, top_vals, alpha=0.15, color='tab:blue')

    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_metric(log: TrainingLog, key: str) -> list[float | None]:
    """Extract a per-epoch metric series from a TrainingLog by key."""
    return [entry.data.get(key) for entry in log.epoch_logs]


def _get_embedding_metric(log: TrainingLog, key: str) -> list[float | None]:
    """Extract a per-epoch metric series from TrainingLog.embedding_model_log."""
    if log.embedding_model_log is None:
        return []
    return [entry.data.get(key) for entry in log.embedding_model_log.epoch_logs]


def _filter_nones(values: list[float | None]) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, values) arrays with None entries removed."""
    valid_indices = [i for i, v in enumerate(values) if v is not None]
    valid_values = np.array([v for v in values if v is not None], dtype=float)
    return np.array(valid_indices, dtype=int), valid_values


def _normalize_loss(vals: np.ndarray) -> np.ndarray:
    """Normalize values so that the first value is 1.0.

    Skips normalization if the first value is 0.
    """
    if len(vals) == 0:
        return vals
    first = vals[0]
    if first == 0:
        return vals
    return vals / first


def _ylim_for_tail(
    series: list[tuple[np.ndarray, np.ndarray]],
    skip_first_n_epochs: int,
) -> tuple[float, float] | None:
    """Compute y-axis limits from plotted values at epoch index >= skip_first_n_epochs.

    Returns (y_min, y_max) with 5% padding, or None if no tail data is available.
    The series argument is a list of (index_array, value_array) pairs as already
    plotted (i.e. log-transformed or otherwise scaled).
    """
    tail: list[float] = []
    for idx, vals in series:
        mask = idx >= skip_first_n_epochs
        if mask.any():
            tail.extend(vals[mask].tolist())
    if not tail:
        return None
    arr = np.array(tail)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    margin = 0.05 * (hi - lo) if hi - lo > 1e-6 else 0.1
    return lo - margin, hi + margin


def _set_log_yticks_with_original_scale(
    axes: plt.Axes,
    log_vals: np.ndarray,
    log_range: tuple[float, float] | None = None,
) -> None:
    """Set y-axis tick labels to show original (pre-log) values on a log-transformed axis.

    Generates 6 evenly-spaced ticks across the data range and labels each with
    the corresponding exponentiated value.  When log_range is provided, ticks are
    placed within that (min, max) range instead of the full data range — use this
    together with axes.set_ylim to keep ticks inside a restricted view.
    """
    if len(log_vals) == 0:
        return
    if log_range is not None:
        min_log, max_log = log_range
    else:
        min_log = float(np.min(log_vals))
        max_log = float(np.max(log_vals))
    if max_log - min_log < 1e-6:
        return
    tick_positions = np.linspace(min_log, max_log, 6)
    tick_labels = [_format_log_tick_value(float(np.exp(t))) for t in tick_positions]
    axes.set_yticks(tick_positions.tolist())
    axes.set_yticklabels(tick_labels)


def _format_log_tick_value(v: float) -> str:
    """Format a de-logged value for a y-axis tick label."""
    if v >= 10000:
        return f"{v:.0f}"
    if v >= 1000:
        return f"{v:.0f}"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    if v >= 0.1:
        return f"{v:.3f}"
    return f"{v:.3g}"


def _rolling_window(n: int) -> int:
    """Return a sensible rolling-mean window size given n data points."""
    return min(_ROLLING_WINDOW, max(1, n // 2))


def _set_accuracy_yticks(axes: plt.Axes, vals: np.ndarray) -> None:
    """Set y-axis ticks on an accuracy axes to show percentage labels."""
    if len(vals) == 0:
        return
    min_acc = float(np.min(vals))
    max_acc = float(np.max(vals))
    accuracy_levels = _generate_accuracy_levels(min_acc, max_acc)
    y_positions = [-np.log(1 - acc + 1e-10) for acc in accuracy_levels]
    y_labels = [f"{acc * 100:.1f}%" for acc in accuracy_levels]
    axes.set_yticks(y_positions)
    axes.set_yticklabels(y_labels)


def _generate_accuracy_levels(min_acc: float, max_acc: float, n_levels: int = 6) -> list[float]:
    """Dynamically generate accuracy tick levels in the pattern 0.5, 0.9, 0.95, 0.99, …

    Returns only levels within or near the [min_acc, max_acc] range.
    """
    min_acc = max(float(min_acc), 0.0)
    max_acc = min(float(max_acc), 1.0)

    if max_acc - min_acc < 1e-6:
        return [min_acc]

    candidates = [0.0, 0.5]
    for num_nines in range(1, 10):
        candidates.append(1.0 - 10 ** (-num_nines))
        candidates.append(1.0 - 0.5 * 10 ** (-num_nines))
    candidates.append(1.0)

    filtered = [level for level in candidates if min_acc <= level <= max_acc]

    if len(filtered) < 2:
        filtered = sorted({min_acc, max_acc})

    if len(filtered) > n_levels:
        step = len(filtered) / (n_levels - 1)
        indices = [int(i * step) for i in range(n_levels - 1)] + [len(filtered) - 1]
        filtered = [filtered[i] for i in sorted(set(indices))]

    return sorted(filtered)
