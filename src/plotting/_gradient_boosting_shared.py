"""Shared plotting helpers for gradient-boosting scoring and length-prediction models.

Provides functions to visualise per-epoch XGBoost diagnostics that are common to
both GradientBoostingModel and GbLengthPredictionModel.
"""

import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting.core import _get_metric, plot_linear_components, plot_loss_components


def plot_block_importance(axes: plt.Axes, log: TrainingLog) -> None:
    """Plot fraction of total XGBoost gain per input feature block over training.

    Reads all ``importance_<block_name>`` keys (excluding per-feature
    ``importance_feature/…`` keys) from the log and plots them as a multi-line
    chart on a linear scale.  The lines sum to 1 at each epoch.
    """
    if not log.epoch_logs:
        axes.set_title("Feature block importance")
        return

    first = log.epoch_logs[0].data
    block_keys = sorted(
        k for k in first
        if k.startswith("importance_") and not k.startswith("importance_feature/")
    )

    if not block_keys:
        axes.set_title("Feature block importance")
        return

    components: dict[str, list[float | None]] = {
        k.removeprefix("importance_"): _get_metric(log, k)
        for k in block_keys
    }
    plot_linear_components(axes, components, "Feature block importance", ylabel="Fraction of total gain")


def plot_convergence_diagnostics(axes: plt.Axes, log: TrainingLog) -> None:
    """Plot tree-level convergence diagnostics on log scale.

    Plots ``train_prediction_std`` (spread of scores across training samples)
    and ``tree_contribution_mean`` (mean absolute delta added by the last tree).
    The latter is only present from round 2 onward.
    """
    components: dict[str, list[float | None]] = {}

    pred_std = _get_metric(log, "train_prediction_std")
    if any(v is not None for v in pred_std):
        components["Prediction std"] = pred_std

    tree_contrib = _get_metric(log, "tree_contribution_mean")
    if any(v is not None for v in tree_contrib):
        components["Tree contribution mean"] = tree_contrib

    plot_loss_components(axes, components, "Convergence diagnostics")
