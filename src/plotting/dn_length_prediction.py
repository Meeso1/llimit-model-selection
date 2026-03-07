import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting.core import (
    plot_combined_loss as _plot_combined_loss,
    plot_combined_accuracy as _plot_combined_accuracy,
    plot_combined_positive_metric as _plot_combined_positive_metric,
    plot_combined_relative_error as _plot_combined_relative_error,
    plot_combined_ratio_around_one as _plot_combined_ratio_around_one,
    _get_metric,
)


def plot_metrics(log: TrainingLog) -> plt.Figure:
    """Create a figure with all metrics for DnEmbeddingLengthPredictionModel.

    Layout: 4 rows × 2 columns (last panel hidden).
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))

    plot_loss(axes[0, 0], log)
    plot_accuracy(axes[0, 1], log)
    plot_mae(axes[1, 0], log)
    plot_rmse(axes[1, 1], log)
    plot_relative_error(axes[2, 0], log)
    plot_relative_ratio(axes[2, 1], log)
    plot_stddev_ratio(axes[3, 0], log)
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


def plot_mae(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_positive_metric(
        axes,
        _get_metric(log, 'train_mae'),
        _get_metric(log, 'val_mae'),
        'Mean Absolute Error (token space)',
        ylabel='MAE (tokens)',
        show_original_scale=True,
    )


def plot_rmse(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_positive_metric(
        axes,
        _get_metric(log, 'train_rmse'),
        _get_metric(log, 'val_rmse'),
        'RMSE (scaled log-space)',
        ylabel='RMSE',
        show_original_scale=True,
    )


def plot_relative_error(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_relative_error(
        axes,
        _get_metric(log, 'train_avg_relative_error'),
        _get_metric(log, 'val_avg_relative_error'),
        'Average Relative Error',
    )


def plot_relative_ratio(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_ratio_around_one(
        axes,
        _get_metric(log, 'train_avg_relative_ratio'),
        _get_metric(log, 'val_avg_relative_ratio'),
        'Average Relative Ratio (ideal = 1.0)',
        ylabel='Ratio',
    )


def plot_stddev_ratio(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_ratio_around_one(
        axes,
        _get_metric(log, 'train_stddev_ratio'),
        _get_metric(log, 'val_stddev_ratio'),
        'Stddev Ratio (ideal = 1.0)',
        ylabel='Ratio',
    )
