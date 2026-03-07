import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting.core import (
    plot_combined_loss as _plot_combined_loss,
    plot_combined_accuracy as _plot_combined_accuracy,
    plot_loss_components as _plot_loss_components,
    plot_accuracy_breakdown as _plot_accuracy_breakdown,
    plot_distribution_over_time as _plot_distribution_over_time,
    _get_metric,
)


def plot_metrics(log: TrainingLog) -> plt.Figure:
    """Create a figure with all metrics for the simple scoring model.

    Layout: 4 rows × 2 columns (last panel hidden).
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))

    plot_loss(axes[0, 0], log)
    plot_accuracy(axes[0, 1], log)
    plot_train_loss_components(axes[1, 0], log)
    plot_val_loss_components(axes[1, 1], log)
    plot_train_accuracy_breakdown(axes[2, 0], log)
    plot_val_accuracy_breakdown(axes[2, 1], log)
    plot_score_distribution(axes[3, 0], log)
    axes[3, 1].set_visible(False)

    fig.tight_layout()
    return fig


def plot_loss(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_loss(
        axes,
        _get_metric(log, 'train_loss'),
        _get_metric(log, 'val_loss'),
        'Loss',
    )


def plot_accuracy(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_accuracy(
        axes,
        _get_metric(log, 'train_accuracy'),
        _get_metric(log, 'val_accuracy'),
        'Accuracy',
    )


def plot_train_loss_components(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_loss_components(
        axes,
        {
            'ranking': _get_metric(log, 'ranking_loss'),
            'tie': _get_metric(log, 'tie_loss'),
            'both_bad': _get_metric(log, 'both_bad_loss'),
        },
        'Loss Components (Training)',
    )


def plot_val_loss_components(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_loss_components(
        axes,
        {
            'ranking': _get_metric(log, 'val_ranking_loss'),
            'tie': _get_metric(log, 'val_tie_loss'),
            'both_bad': _get_metric(log, 'val_both_bad_loss'),
        },
        'Loss Components (Validation)',
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


def plot_score_distribution(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_distribution_over_time(
        axes,
        _get_metric(log, 'avg_score'),
        _get_metric(log, 'top_10_pct_score'),
        _get_metric(log, 'bottom_10_pct_score'),
        'Score Distribution',
        ylabel='Score',
    )
