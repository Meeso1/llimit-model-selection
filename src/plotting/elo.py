import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting.core import (
    plot_combined_accuracy as _plot_combined_accuracy,
    plot_accuracy_breakdown as _plot_accuracy_breakdown,
    plot_distribution_over_time as _plot_distribution_over_time,
    plot_delta_metric as _plot_delta_metric,
    _get_metric,
)


def plot_metrics(log: TrainingLog) -> plt.Figure:
    """Create a figure with all metrics for the ELO scoring model.

    Layout: 3 rows × 2 columns (last panel hidden).
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    plot_accuracy(axes[0, 0], log)
    plot_train_accuracy_breakdown(axes[0, 1], log)
    plot_val_accuracy_breakdown(axes[1, 0], log)
    plot_rating_distribution(axes[1, 1], log)
    plot_rating_change(axes[2, 0], log)
    axes[2, 1].set_visible(False)

    fig.tight_layout()
    return fig


def plot_accuracy(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_accuracy(
        axes,
        _get_metric(log, 'train_accuracy'),
        _get_metric(log, 'val_accuracy'),
        'Accuracy',
    )


def plot_train_accuracy_breakdown(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_accuracy_breakdown(
        axes,
        {
            'ranking': _get_metric(log, 'ranking_accuracy'),
            'tie': _get_metric(log, 'tie_accuracy'),
            'both_bad': _get_metric(log, 'both_bad_accuracy'),
        },
        'Accuracy Breakdown (Training)',
    )


def plot_val_accuracy_breakdown(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_accuracy_breakdown(
        axes,
        {
            'ranking': _get_metric(log, 'val_ranking_accuracy'),
            'tie': _get_metric(log, 'val_tie_accuracy'),
            'both_bad': _get_metric(log, 'val_both_bad_accuracy'),
        },
        'Accuracy Breakdown (Validation)',
    )


def plot_rating_distribution(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_distribution_over_time(
        axes,
        _get_metric(log, 'avg_rating'),
        _get_metric(log, 'top_10_pct_rating'),
        _get_metric(log, 'bottom_10_pct_rating'),
        'Rating Distribution',
        ylabel='ELO Rating',
    )


def plot_rating_change(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_delta_metric(
        axes,
        _get_metric(log, 'avg_rating_change'),
        'Average Rating Change per Epoch',
        ylabel='Δ Rating',
    )
