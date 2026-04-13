"""Shared plotting helpers for length prediction models.

Used by both ``dn_length_prediction`` and ``gb_length_prediction``; not
intended for direct import by callers — use the model-specific modules instead.
"""

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
