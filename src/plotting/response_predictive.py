import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting.core import (
    plot_loss as _plot_loss,
    plot_combined_loss as _plot_combined_loss,
    plot_combined_accuracy as _plot_combined_accuracy,
    plot_combined_log_variance as _plot_combined_log_variance,
    plot_loss_components as _plot_loss_components,
    _get_metric,
    _filter_nones,
)


def _weighted_metric(
    values: list[float | None],
    weight: float,
) -> list[float | None]:
    """Multiply each non-None value by weight."""
    return [v * weight if v is not None else None for v in values]


def plot_metrics(log: TrainingLog) -> plt.Figure:
    """Create a figure with all metrics for the response predictive model.

    Layout: 6 rows × 2 columns.
    """
    fig, axes = plt.subplots(6, 2, figsize=(14, 30))

    plot_total_loss(axes[0, 0], log)
    plot_accuracy(axes[0, 1], log)
    plot_scorer_real_repr_accuracy(axes[1, 0], log)
    plot_prediction_quality(axes[1, 1], log)
    plot_repr_mean_variance(axes[2, 0], log)
    plot_predictability_loss(axes[2, 1], log)
    plot_repr_kl_loss(axes[3, 0], log)
    plot_scoring_loss(axes[3, 1], log)
    plot_prediction_loss(axes[4, 0], log)
    plot_real_repr_ratio(axes[4, 1], log)
    plot_component_losses_weighted(axes[5, 0], log)
    plot_component_losses_normalized(axes[5, 1], log)

    fig.tight_layout()
    return fig


def plot_total_loss(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_loss(
        axes,
        _get_metric(log, 'train_loss'),
        _get_metric(log, 'val_loss'),
        'Total Loss',
    )


def plot_accuracy(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_accuracy(
        axes,
        _get_metric(log, 'train_accuracy'),
        _get_metric(log, 'val_accuracy'),
        'Accuracy',
    )


def plot_scorer_real_repr_accuracy(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_accuracy(
        axes,
        _get_metric(log, 'scorer_real_repr_accuracy'),
        _get_metric(log, 'val_scorer_real_repr_accuracy'),
        'Scorer Real Representation Accuracy',
    )


def plot_prediction_quality(axes: plt.Axes, log: TrainingLog) -> None:
    """Prediction quality does not have a monotonic target direction."""
    _plot_combined_accuracy(
        axes,
        _get_metric(log, 'prediction_quality'),
        _get_metric(log, 'val_prediction_quality'),
        'Prediction Quality',
        mark_best_val=False,
    )


def plot_repr_mean_variance(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_log_variance(
        axes,
        _get_metric(log, 'repr_mean_variance'),
        _get_metric(log, 'val_repr_mean_variance'),
        'Response Representation Variance',
    )


def plot_predictability_loss(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_loss(
        axes,
        _get_metric(log, 'predictability_loss'),
        _get_metric(log, 'val_predictability_loss'),
        'Predictability Loss',
    )


def plot_repr_kl_loss(axes: plt.Axes, log: TrainingLog) -> None:
    """KL loss does not necessarily decrease - it measures representation diversity."""
    _plot_combined_loss(
        axes,
        _get_metric(log, 'repr_kl_loss'),
        _get_metric(log, 'val_repr_kl_loss'),
        'Representation KL Loss',
        mark_best_val=False,
    )


def plot_scoring_loss(axes: plt.Axes, log: TrainingLog) -> None:
    """Training-only metric."""
    _plot_loss(axes, _get_metric(log, 'scoring_loss'), 'Scoring Loss (Training)')


def plot_prediction_loss(axes: plt.Axes, log: TrainingLog) -> None:
    """Training-only metric."""
    _plot_loss(axes, _get_metric(log, 'prediction_loss'), 'Prediction Loss (Training)')


def plot_real_repr_ratio(axes: plt.Axes, log: TrainingLog) -> None:
    """Curriculum schedule - fraction of real representations used during training."""
    idx, vals = _filter_nones(_get_metric(log, 'current_real_repr_ratio'))
    if len(vals) == 0:
        axes.set_visible(False)
        return
    axes.plot(idx, vals)
    axes.set_ylim(0, 1)
    axes.set_ylabel("Ratio")
    axes.set_title("Real Representation Ratio (curriculum)")


def plot_component_losses_weighted(axes: plt.Axes, log: TrainingLog) -> None:
    """Training component losses scaled by their actual loss weights.

    Shows the true gradient contribution of each component (i.e. what the optimizer
    actually sees), which is the most direct view for diagnosing weight imbalances.
    Weights are read from the training log config; defaults match the model constructor.
    """
    pw = float(log.config.get('prediction_loss_weight', 1.0))
    plw = float(log.config.get('predictability_loss_weight', 0.2))
    kw = float(log.config.get('repr_kl_loss_weight', 0.01))

    _plot_loss_components(
        axes,
        {
            f'Scoring (×1)': _get_metric(log, 'scoring_loss'),
            f'Prediction (×{pw:g})': _weighted_metric(_get_metric(log, 'prediction_loss'), pw),
            f'Predictability (×{plw:g})': _weighted_metric(_get_metric(log, 'predictability_loss'), plw),
            f'KL (×{kw:g})': _weighted_metric(_get_metric(log, 'repr_kl_loss'), kw),
        },
        'Weighted Component Losses (Training)',
        normalize=False,
    )


def plot_component_losses_normalized(axes: plt.Axes, log: TrainingLog) -> None:
    """Weighted training component losses each normalized to start at 1.

    Shows relative improvement trajectories — which component is still learning
    and which has plateaued — independent of absolute scale.
    """
    pw = float(log.config.get('prediction_loss_weight', 1.0))
    plw = float(log.config.get('predictability_loss_weight', 0.2))
    kw = float(log.config.get('repr_kl_loss_weight', 0.01))

    _plot_loss_components(
        axes,
        {
            f'Scoring (×1)': _get_metric(log, 'scoring_loss'),
            f'Prediction (×{pw:g})': _weighted_metric(_get_metric(log, 'prediction_loss'), pw),
            f'Predictability (×{plw:g})': _weighted_metric(_get_metric(log, 'predictability_loss'), plw),
            f'KL (×{kw:g})': _weighted_metric(_get_metric(log, 'repr_kl_loss'), kw),
        },
        'Weighted Component Losses Normalized (Training)',
        normalize=True,
    )
