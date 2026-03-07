import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting.core import (
    plot_combined_loss as _plot_combined_loss,
    plot_combined_accuracy as _plot_combined_accuracy,
    _get_metric,
)


def plot_metrics(log: TrainingLog) -> plt.Figure:
    """Create a figure with all metrics for DenseNetworkModel.

    Layout: 1 row × 2 columns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_loss(axes[0], log)
    plot_accuracy(axes[1], log)

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
