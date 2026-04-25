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

    Layout: 2 rows × 2 columns (bottom-right panel hidden).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_total_loss(axes[0, 0], log)
    plot_universal_accuracy(axes[0, 1], log)
    plot_loss_components(axes[1, 0], log)
    axes[1, 1].set_visible(False)

    fig.tight_layout()
    return fig


def plot_total_loss(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_loss(
        axes,
        _get_embedding_metric(log, "train_loss"),
        _get_embedding_metric(log, "val_loss"),
        "Triplet embedding — total loss",
    )


def plot_universal_accuracy(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_accuracy(
        axes,
        _get_embedding_metric(log, "train_universal_accuracy"),
        _get_embedding_metric(log, "val_universal_accuracy"),
        "Triplet embedding — universal accuracy",
    )


def plot_loss_components(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_loss_components(
        axes,
        {
            "Triplet": _get_embedding_metric(log, "train_triplet_loss"),
            "Regularization": _get_embedding_metric(log, "train_reg_loss"),
        },
        "Triplet embedding — loss components (train)",
        normalize=True,
    )
