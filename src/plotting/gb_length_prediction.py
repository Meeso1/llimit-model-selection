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
from src.plotting._gradient_boosting_shared import plot_block_importance, plot_convergence_diagnostics


def plot_metrics(log: TrainingLog, skip_first_n_epochs: int = 0) -> plt.Figure:
    """Create a figure with all metrics for GbLengthPredictionModel.

    When skip_first_n_epochs > 0, the y-axis range of metric plots is computed
    from that epoch onward (all epochs are still drawn).

    Layout: 5 rows × 2 columns.
    """
    fig, axes = plt.subplots(5, 2, figsize=(14, 25))

    plot_loss(axes[0, 0], log)
    plot_accuracy(axes[0, 1], log)
    plot_mae(axes[1, 0], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_rmse(axes[1, 1], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_relative_error(axes[2, 0], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_relative_ratio(axes[2, 1], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_stddev_ratio(axes[3, 0], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_convergence_diagnostics(axes[3, 1], log)
    plot_block_importance(axes[4, 0], log)
    axes[4, 1].set_visible(False)

    fig.tight_layout()
    return fig
