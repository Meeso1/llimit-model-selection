import matplotlib.pyplot as plt
import numpy as np

from src.utils.training_logger import TrainingLog


_ROLLING_WINDOW = 10


def plot_loss(axes: plt.Axes, values: list[float | None], title: str) -> None:
    """Plot a single loss series with log scale and rolling mean."""
    idx, vals = _filter_nones(values)
    if len(vals) == 0:
        axes.set_title(title)
        return

    log_loss = np.log(vals)
    window = _rolling_window(len(vals))

    axes.plot(idx, log_loss, label="log(loss)", alpha=0.6)

    if window >= 2:
        sliding_mean = np.convolve(log_loss, np.ones(window) / window, mode='valid')
        axes.plot(idx[window - 1:], sliding_mean, label=f"Rolling mean ({window})")
        min_val = np.min(sliding_mean)
        axes.axhline(y=min_val, color='r', linestyle='--', label=f'Min: {min_val:.3f}')

    axes.set_ylabel("log(loss)")
    axes.set_title(title)
    axes.legend()


def plot_accuracy(axes: plt.Axes, values: list[float | None], title: str) -> None:
    """Plot a single accuracy series with -log(1-acc) scale and rolling mean."""
    idx, vals = _filter_nones(values)
    if len(vals) == 0:
        axes.set_title(title)
        return

    inv_log_acc = -np.log(1 - vals + 1e-10)
    window = _rolling_window(len(vals))

    axes.plot(idx, inv_log_acc, label="-log(1-accuracy)", alpha=0.6)

    if window >= 2:
        sliding_mean = np.convolve(inv_log_acc, np.ones(window) / window, mode='valid')
        axes.plot(idx[window - 1:], sliding_mean, label=f"Rolling mean ({window})")
        max_val = np.max(sliding_mean)
        max_accuracy = 1 - np.exp(-max_val)
        axes.axhline(y=max_val, color='r', linestyle='--', label=f'Max: {max_accuracy * 100:.1f}%')

    _set_accuracy_yticks(axes, vals)
    axes.set_ylabel("Accuracy")
    axes.set_title(title)
    axes.legend()


def plot_train_val_loss(
    axes: plt.Axes,
    train_values: list[float | None],
    val_values: list[float | None],
    title: str,
) -> None:
    """Plot training and validation loss together on log scale. No rolling mean."""
    train_idx, train_vals = _filter_nones(train_values)
    val_idx, val_vals = _filter_nones(val_values)

    if len(train_vals) == 0 and len(val_vals) == 0:
        axes.set_title(title)
        return

    if len(train_vals) > 0:
        axes.plot(train_idx, np.log(np.maximum(train_vals, 1e-10)), label="Train", alpha=0.8)
    if len(val_vals) > 0:
        axes.plot(val_idx, np.log(np.maximum(val_vals, 1e-10)), label="Val", alpha=0.8)

    axes.set_ylabel("log(loss)")
    axes.set_title(title)
    axes.legend()


def plot_train_val_accuracy(
    axes: plt.Axes,
    train_values: list[float | None],
    val_values: list[float | None],
    title: str,
) -> None:
    """Plot training and validation accuracy together with -log(1-acc) scale. No rolling mean."""
    train_idx, train_vals = _filter_nones(train_values)
    val_idx, val_vals = _filter_nones(val_values)

    if len(train_vals) == 0 and len(val_vals) == 0:
        axes.set_title(title)
        return

    if len(train_vals) > 0:
        axes.plot(train_idx, -np.log(1 - train_vals + 1e-10), label="Train", alpha=0.8)
    if len(val_vals) > 0:
        axes.plot(val_idx, -np.log(1 - val_vals + 1e-10), label="Val", alpha=0.8)

    all_vals = np.concatenate([train_vals, val_vals])
    _set_accuracy_yticks(axes, all_vals)
    axes.set_ylabel("Accuracy")
    axes.set_title(title)
    axes.legend()


def plot_train_val_positive_metric(
    axes: plt.Axes,
    train_values: list[float | None],
    val_values: list[float | None],
    title: str,
    ylabel: str = "log(value)",
) -> None:
    """Plot train and validation of any positive metric on log scale. No rolling mean.

    Suitable for MAE, RMSE, or similar metrics where log scale aids readability.
    """
    train_idx, train_vals = _filter_nones(train_values)
    val_idx, val_vals = _filter_nones(val_values)

    if len(train_vals) == 0 and len(val_vals) == 0:
        axes.set_title(title)
        return

    if len(train_vals) > 0:
        axes.plot(train_idx, np.log(np.maximum(train_vals, 1e-10)), label="Train", alpha=0.8)
    if len(val_vals) > 0:
        axes.plot(val_idx, np.log(np.maximum(val_vals, 1e-10)), label="Val", alpha=0.8)

    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()


def plot_train_val_relative_error(
    axes: plt.Axes,
    train_values: list[float | None],
    val_values: list[float | None],
    title: str,
) -> None:
    """Plot train and validation relative error together using log(1+error) scale. No rolling mean.

    The log(1 + error) transform expands the scale near 0 where improvements matter most.
    """
    train_idx, train_vals = _filter_nones(train_values)
    val_idx, val_vals = _filter_nones(val_values)

    if len(train_vals) == 0 and len(val_vals) == 0:
        axes.set_title(title)
        return

    if len(train_vals) > 0:
        axes.plot(train_idx, np.log1p(train_vals), label="Train", alpha=0.8)
    if len(val_vals) > 0:
        axes.plot(val_idx, np.log1p(val_vals), label="Val", alpha=0.8)

    axes.set_ylabel("log(1 + relative error)")
    axes.set_title(title)
    axes.legend()


def plot_train_val_ratio_around_one(
    axes: plt.Axes,
    train_values: list[float | None],
    val_values: list[float | None],
    title: str,
    ylabel: str = "Ratio",
) -> None:
    """Plot train and validation of a ratio that should ideally be 1.0. No rolling mean.

    Displays a horizontal reference line at 1.0.
    Suitable for stddev_ratio, avg_relative_ratio.
    """
    train_idx, train_vals = _filter_nones(train_values)
    val_idx, val_vals = _filter_nones(val_values)

    if len(train_vals) == 0 and len(val_vals) == 0:
        axes.set_title(title)
        return

    if len(train_vals) > 0:
        axes.plot(train_idx, train_vals, label="Train", alpha=0.8)
    if len(val_vals) > 0:
        axes.plot(val_idx, val_vals, label="Val", alpha=0.8)

    axes.axhline(y=1.0, color='g', linestyle='--', label='Ideal (1.0)', alpha=0.7)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()


def plot_train_val_log_variance(
    axes: plt.Axes,
    train_values: list[float | None],
    val_values: list[float | None],
    title: str,
) -> None:
    """Plot train and validation variance in log scale. No rolling mean."""
    train_idx, train_vals = _filter_nones(train_values)
    val_idx, val_vals = _filter_nones(val_values)

    if len(train_vals) == 0 and len(val_vals) == 0:
        axes.set_title(title)
        return

    if len(train_vals) > 0:
        axes.plot(train_idx, np.log(np.maximum(train_vals, 1e-10)), label="Train", alpha=0.8)
    if len(val_vals) > 0:
        axes.plot(val_idx, np.log(np.maximum(val_vals, 1e-10)), label="Val", alpha=0.8)

    axes.set_ylabel("log(variance)")
    axes.set_title(title)
    axes.legend()


def plot_positive_metric(
    axes: plt.Axes,
    values: list[float | None],
    title: str,
    ylabel: str = "log(value)",
) -> None:
    """Plot a single positive metric with log scale and rolling mean.

    Suitable for MAE, RMSE, or any positive-valued metric that decreases over training.
    """
    idx, vals = _filter_nones(values)
    if len(vals) == 0:
        axes.set_title(title)
        return

    log_vals = np.log(np.maximum(vals, 1e-10))
    window = _rolling_window(len(vals))

    axes.plot(idx, log_vals, label="log(value)", alpha=0.6)

    if window >= 2:
        rolling = np.convolve(log_vals, np.ones(window) / window, mode='valid')
        axes.plot(idx[window - 1:], rolling, label=f"Rolling mean ({window})")
        min_val = np.min(rolling)
        axes.axhline(y=min_val, color='r', linestyle='--', label=f'Min: {np.exp(min_val):.4g}')

    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()


def plot_relative_error(
    axes: plt.Axes,
    values: list[float | None],
    title: str,
) -> None:
    """Plot relative error with log(1+error) scale and rolling mean.

    Lower is better. The log(1+error) transform expands the scale near 0.
    """
    idx, vals = _filter_nones(values)
    if len(vals) == 0:
        axes.set_title(title)
        return

    log_err = np.log1p(vals)
    window = _rolling_window(len(vals))

    axes.plot(idx, log_err, label="log(1 + error)", alpha=0.6)

    if window >= 2:
        rolling = np.convolve(log_err, np.ones(window) / window, mode='valid')
        axes.plot(idx[window - 1:], rolling, label=f"Rolling mean ({window})")
        min_rolling = np.min(rolling)
        axes.axhline(y=min_rolling, color='r', linestyle='--', label=f'Min: {np.expm1(min_rolling):.3f}')

    axes.set_ylabel("log(1 + relative error)")
    axes.set_title(title)
    axes.legend()


def plot_ratio_around_one(
    axes: plt.Axes,
    values: list[float | None],
    title: str,
    ylabel: str = "Ratio",
) -> None:
    """Plot a ratio metric that should ideally be 1.0 with rolling mean.

    Shows a reference line at 1.0. Suitable for stddev_ratio, avg_relative_ratio.
    """
    idx, vals = _filter_nones(values)
    if len(vals) == 0:
        axes.set_title(title)
        return

    window = _rolling_window(len(vals))

    axes.plot(idx, vals, label="value", alpha=0.6)

    if window >= 2:
        rolling = np.convolve(vals, np.ones(window) / window, mode='valid')
        axes.plot(idx[window - 1:], rolling, label=f"Rolling mean ({window})")

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

    Shows a reference line at 0.
    """
    idx, vals = _filter_nones(values)
    if len(vals) == 0:
        axes.set_title(title)
        return

    window = _rolling_window(len(vals))

    axes.plot(idx, vals, label="value", alpha=0.6)

    if window >= 2:
        rolling = np.convolve(vals, np.ones(window) / window, mode='valid')
        axes.plot(idx[window - 1:], rolling, label=f"Rolling mean ({window})")

    axes.axhline(y=0.0, color='g', linestyle='--', label='Zero', alpha=0.7)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()


def plot_loss_components(
    axes: plt.Axes,
    components: dict[str, list[float | None]],
    title: str,
) -> None:
    """Plot multiple loss components on the same axes in log scale. No rolling mean."""
    has_data = False
    for name, values in components.items():
        idx, vals = _filter_nones(values)
        if len(vals) > 0:
            axes.plot(idx, np.log(np.maximum(vals, 1e-10)), label=name, alpha=0.8)
            has_data = True

    if not has_data:
        axes.set_title(title)
        return

    axes.set_ylabel("log(loss)")
    axes.set_title(title)
    axes.legend()


def plot_accuracy_breakdown(
    axes: plt.Axes,
    metrics: dict[str, list[float | None]],
    title: str,
) -> None:
    """Plot multiple accuracy components together with -log(1-acc) scale. No rolling mean.

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


def plot_length_prediction_history(log: TrainingLog) -> plt.Figure:
    """Create a complete figure summarising length prediction model training history.

    Layout: 4 rows × 2 columns covering loss, accuracy, MAE, RMSE,
    relative error, relative ratio, and stddev ratio.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))

    plot_train_val_loss(axes[0, 0], _get_metric(log, "train_loss"), _get_metric(log, "val_loss"), "Loss")
    plot_train_val_accuracy(axes[0, 1], _get_metric(log, "train_accuracy"), _get_metric(log, "val_accuracy"), "Accuracy")

    plot_train_val_positive_metric(
        axes[1, 0],
        _get_metric(log, "train_mae"),
        _get_metric(log, "val_mae"),
        "Mean Absolute Error (token space)",
        ylabel="log(MAE)",
    )
    plot_train_val_positive_metric(
        axes[1, 1],
        _get_metric(log, "train_rmse"),
        _get_metric(log, "val_rmse"),
        "RMSE (scaled log-space)",
        ylabel="log(RMSE)",
    )

    plot_train_val_relative_error(
        axes[2, 0],
        _get_metric(log, "train_avg_relative_error"),
        _get_metric(log, "val_avg_relative_error"),
        "Average Relative Error",
    )
    plot_train_val_ratio_around_one(
        axes[2, 1],
        _get_metric(log, "train_avg_relative_ratio"),
        _get_metric(log, "val_avg_relative_ratio"),
        "Average Relative Ratio (ideal = 1.0)",
        ylabel="Ratio",
    )

    plot_train_val_ratio_around_one(
        axes[3, 0],
        _get_metric(log, "train_stddev_ratio"),
        _get_metric(log, "val_stddev_ratio"),
        "Stddev Ratio (ideal = 1.0)",
        ylabel="Ratio",
    )
    axes[3, 1].set_visible(False)

    fig.suptitle("Length Prediction – Training History", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_response_predictive_history(log: TrainingLog) -> plt.Figure:
    """Create a complete figure summarising response predictive model training history.

    Layout: 5 rows × 2 columns covering loss, accuracy, representation metrics,
    component losses, and curriculum schedule.
    """
    fig, axes = plt.subplots(5, 2, figsize=(14, 25))

    plot_train_val_loss(axes[0, 0], _get_metric(log, "train_loss"), _get_metric(log, "val_loss"), "Total Loss")
    plot_train_val_accuracy(axes[0, 1], _get_metric(log, "train_accuracy"), _get_metric(log, "val_accuracy"), "Accuracy")

    plot_train_val_accuracy(
        axes[1, 0],
        _get_metric(log, "scorer_real_repr_accuracy"),
        _get_metric(log, "val_scorer_real_repr_accuracy"),
        "Scorer Real Representation Accuracy",
    )
    plot_train_val_accuracy(
        axes[1, 1],
        _get_metric(log, "prediction_quality"),
        _get_metric(log, "val_prediction_quality"),
        "Prediction Quality",
    )

    plot_train_val_log_variance(
        axes[2, 0],
        _get_metric(log, "repr_mean_variance"),
        _get_metric(log, "val_repr_mean_variance"),
        "Response Representation Variance",
    )
    plot_train_val_loss(
        axes[2, 1],
        _get_metric(log, "predictability_loss"),
        _get_metric(log, "val_predictability_loss"),
        "Predictability Loss",
    )

    plot_train_val_loss(
        axes[3, 0],
        _get_metric(log, "repr_kl_loss"),
        _get_metric(log, "val_repr_kl_loss"),
        "Representation KL Loss",
    )
    plot_loss(axes[3, 1], _get_metric(log, "scoring_loss"), "Scoring Loss (Train)")

    plot_loss(axes[4, 0], _get_metric(log, "prediction_loss"), "Prediction Loss (Train)")

    idx, vals = _filter_nones(_get_metric(log, "current_real_repr_ratio"))
    if len(vals) > 0:
        axes[4, 1].plot(idx, vals)
        axes[4, 1].set_ylim(0, 1)
        axes[4, 1].set_ylabel("Ratio")
        axes[4, 1].set_title("Real Representation Ratio (curriculum)")
    else:
        axes[4, 1].set_visible(False)

    fig.suptitle("Response Predictive Model – Training History", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_simple_scoring_history(log: TrainingLog) -> plt.Figure:
    """Create a complete figure summarising simple scoring model training history.

    Layout: 4 rows × 2 columns covering loss, accuracy, loss components,
    accuracy breakdown, and score distribution.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))

    plot_train_val_loss(axes[0, 0], _get_metric(log, "train_loss"), _get_metric(log, "val_loss"), "Loss")
    plot_train_val_accuracy(axes[0, 1], _get_metric(log, "train_accuracy"), _get_metric(log, "val_accuracy"), "Accuracy")

    plot_loss_components(
        axes[1, 0],
        {
            "ranking": _get_metric(log, "ranking_loss"),
            "tie": _get_metric(log, "tie_loss"),
            "both_bad": _get_metric(log, "both_bad_loss"),
        },
        "Loss Components (Train)",
    )
    plot_loss_components(
        axes[1, 1],
        {
            "ranking": _get_metric(log, "val_ranking_loss"),
            "tie": _get_metric(log, "val_tie_loss"),
            "both_bad": _get_metric(log, "val_both_bad_loss"),
        },
        "Loss Components (Val)",
    )

    plot_accuracy_breakdown(
        axes[2, 0],
        {
            "ranking": _get_metric(log, "ranking_accuracy"),
            "tie": _get_metric(log, "tie_accuracy"),
            "both_bad": _get_metric(log, "both_bad_accuracy"),
        },
        "Accuracy Breakdown (Train)",
    )
    plot_accuracy_breakdown(
        axes[2, 1],
        {
            "ranking": _get_metric(log, "val_ranking_accuracy"),
            "tie": _get_metric(log, "val_tie_accuracy"),
            "both_bad": _get_metric(log, "val_both_bad_accuracy"),
        },
        "Accuracy Breakdown (Val)",
    )

    plot_distribution_over_time(
        axes[3, 0],
        _get_metric(log, "avg_score"),
        _get_metric(log, "top_10_pct_score"),
        _get_metric(log, "bottom_10_pct_score"),
        "Score Distribution",
        ylabel="Score",
    )
    axes[3, 1].set_visible(False)

    fig.suptitle("Simple Scoring Model – Training History", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_elo_history(log: TrainingLog) -> plt.Figure:
    """Create a complete figure summarising ELO scoring model training history.

    Layout: 3 rows × 2 columns covering accuracy, accuracy breakdown,
    rating distribution, and per-epoch rating change.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    plot_train_val_accuracy(axes[0, 0], _get_metric(log, "train_accuracy"), _get_metric(log, "val_accuracy"), "Accuracy")

    plot_accuracy_breakdown(
        axes[0, 1],
        {
            "ranking": _get_metric(log, "ranking_accuracy"),
            "tie": _get_metric(log, "tie_accuracy"),
            "both_bad": _get_metric(log, "both_bad_accuracy"),
        },
        "Accuracy Breakdown (Train)",
    )

    plot_accuracy_breakdown(
        axes[1, 0],
        {
            "ranking": _get_metric(log, "val_ranking_accuracy"),
            "tie": _get_metric(log, "val_tie_accuracy"),
            "both_bad": _get_metric(log, "val_both_bad_accuracy"),
        },
        "Accuracy Breakdown (Val)",
    )
    plot_distribution_over_time(
        axes[1, 1],
        _get_metric(log, "avg_rating"),
        _get_metric(log, "top_10_pct_rating"),
        _get_metric(log, "bottom_10_pct_rating"),
        "Rating Distribution",
        ylabel="ELO Rating",
    )

    plot_delta_metric(
        axes[2, 0],
        _get_metric(log, "avg_rating_change"),
        "Average Rating Change per Epoch",
        ylabel="Δ Rating",
    )
    axes[2, 1].set_visible(False)

    fig.suptitle("ELO Scoring Model – Training History", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_metric(log: TrainingLog, key: str) -> list[float | None]:
    """Extract a per-epoch metric series from a TrainingLog by key."""
    return [entry.data.get(key) for entry in log.epoch_logs]


def _filter_nones(values: list[float | None]) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, values) arrays with None entries removed."""
    valid_indices = [i for i, v in enumerate(values) if v is not None]
    valid_values = np.array([v for v in values if v is not None], dtype=float)
    return np.array(valid_indices, dtype=int), valid_values


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
