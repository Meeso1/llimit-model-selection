"""Plotting utilities for AttentionEmbeddingModel training logs.

Triplet and nearest-neighbour accuracies are intentionally omitted from
``plot_metrics`` — they are noisy / weakly indicative in practice.
"""

import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting.core import (
    plot_combined_accuracy as _plot_combined_accuracy,
    plot_combined_loss as _plot_combined_loss,
    _get_embedding_metric,
)


def plot_metrics(log: TrainingLog) -> plt.Figure:
    """Create a figure with all training metrics for AttentionEmbeddingModel.

    Layout: 1 row × 2 columns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_total_loss(axes[0], log)
    plot_universal_accuracy(axes[1], log)

    fig.tight_layout()
    return fig


def plot_total_loss(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_loss(
        axes,
        _get_embedding_metric(log, "train_loss"),
        _get_embedding_metric(log, "val_loss"),
        "Loss",
    )


def plot_universal_accuracy(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_accuracy(
        axes,
        _get_embedding_metric(log, "train_universal_accuracy"),
        _get_embedding_metric(log, "val_universal_accuracy"),
        "Distinguishability accuracy",
    )
