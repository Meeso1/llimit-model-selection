"""Plotting utilities for triplet-based embedding model training logs.

Covers ``TripletFrozenEncoderModel`` and ``TripletFinetunableEncoderModel``.
Triplet and nearest-neighbour accuracies are intentionally omitted from
``plot_metrics`` — they are noisy / weakly indicative in practice.
"""

import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting.core import (
    plot_combined_accuracy as _plot_combined_accuracy,
    plot_combined_loss as _plot_combined_loss,
    plot_loss_components as _plot_loss_components,
    _get_embedding_metric,
)


def plot_metrics(log: TrainingLog) -> plt.Figure:
    """Create a figure with all training metrics for triplet embedding models.

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


def plot_loss_components(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_loss_components(
        axes,
        {
            "Triplet": _get_embedding_metric(log, "train_triplet_loss"),
            "Regularization": _get_embedding_metric(log, "train_reg_loss"),
        },
        "Loss components (train)",
        normalize=True,
    )
