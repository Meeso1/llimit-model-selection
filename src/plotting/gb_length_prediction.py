import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting._length_prediction_shared import (
    plot_loss,
    plot_accuracy,
    plot_mae,
    plot_rmse,
    plot_relative_error,
    plot_relative_ratio,
    plot_stddev_ratio,
)


def plot_metrics(log: TrainingLog) -> plt.Figure:
    """Create a figure with all metrics for GbLengthPredictionModel.

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
