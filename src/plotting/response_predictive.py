import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting.core import (
    plot_loss as _plot_loss,
    plot_combined_loss as _plot_combined_loss,
    plot_combined_accuracy as _plot_combined_accuracy,
    plot_combined_log_variance as _plot_combined_log_variance,
    plot_loss_components as _plot_loss_components,
    plot_score_variance_decomposition as _plot_score_variance_decomposition,
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
    """Create a figure with all per-epoch metrics for the response predictive model.

    Post-training-only metrics (e.g. sensitivity ablations in ``final_metrics``) are
    not plotted.

    Layout: 9 rows × 2 columns.
    """
    fig, axes = plt.subplots(10, 2, figsize=(14, 42))

    plot_total_loss(axes[0, 0], log)
    plot_accuracy(axes[0, 1], log)
    plot_scorer_real_repr_accuracy(axes[1, 0], log)
    plot_prediction_quality(axes[1, 1], log)
    plot_repr_mean_variance(axes[2, 0], log)
    plot_predictability_loss(axes[2, 1], log)
    plot_repr_kl_loss(axes[3, 0], log)
    plot_scoring_loss_breakdown(axes[3, 1], log)
    plot_prediction_loss(axes[4, 0], log)
    plot_score_consistency_loss(axes[4, 1], log)
    plot_repr_dist_kl_loss(axes[5, 0], log)
    plot_component_losses_weighted(axes[5, 1], log)
    plot_component_losses_normalized(axes[6, 0], log)
    plot_diagnostic_grad_norms(axes[6, 1], log)
    plot_predictor_input_proj_norms(axes[7, 0], log)
    plot_pred_real_repr_diagnostics(axes[7, 1], log)
    plot_score_variance_diagnostics(axes[8, 0], log)
    plot_grad_attr_embeddings(axes[8, 1], log)

    fig.tight_layout()
    return fig


def plot_metrics_2x4by2(log: TrainingLog) -> tuple[plt.Figure, plt.Figure]:
    """Create 2 figures with selected metrics for the response predictive model.

    Figure 1 layout (4 rows × 2 columns): training quality and diagnostic signals.
    Figure 2 layout (4 rows × 2 columns): all individual loss components + weighted summary.
    """
    fig1, axes1 = plt.subplots(4, 2, figsize=(14, 20))
    fig2, axes2 = plt.subplots(4, 2, figsize=(14, 20))

    plot_total_loss(axes1[0, 0], log)
    plot_accuracy(axes1[0, 1], log)
    plot_scorer_real_repr_accuracy(axes1[1, 0], log)
    plot_prediction_quality(axes1[1, 1], log)
    plot_repr_mean_variance(axes1[2, 0], log)
    plot_score_variance_diagnostics(axes1[2, 1], log)
    plot_diagnostic_grad_norms(axes1[3, 0], log)
    plot_predictor_input_proj_norms(axes1[3, 1], log)

    plot_scoring_loss_breakdown(axes2[0, 0], log)
    plot_prediction_loss(axes2[0, 1], log)
    plot_predictability_loss(axes2[1, 0], log)
    plot_repr_kl_loss(axes2[1, 1], log)
    plot_score_consistency_loss(axes2[2, 0], log)
    plot_repr_dist_kl_loss(axes2[2, 1], log)
    plot_component_losses_weighted(axes2[3, 0], log)
    axes2[3, 1].set_visible(False)

    fig1.tight_layout()
    fig2.tight_layout()
    return fig1, fig2


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
        'Accuracy (Predicted Representations)',
    )


def plot_scorer_real_repr_accuracy(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_combined_accuracy(
        axes,
        _get_metric(log, 'scorer_real_repr_accuracy'),
        _get_metric(log, 'val_scorer_real_repr_accuracy'),
        'Scorer Accuracy (Real Representations)',
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
        'Prediction / Predictability Loss',
    )


def plot_repr_kl_loss(axes: plt.Axes, log: TrainingLog) -> None:
    """KL loss does not necessarily decrease - it measures representation diversity."""
    _plot_combined_loss(
        axes,
        _get_metric(log, 'repr_kl_loss'),
        _get_metric(log, 'val_repr_kl_loss'),
        'Representation KL Loss (Encoder → N(0,1))',
        mark_best_val=False,
    )


def plot_scoring_loss_breakdown(axes: plt.Axes, log: TrainingLog) -> None:
    """Real-repr vs predicted-repr ranking losses (training only).

    Shows whether the two scoring paths are balanced and whether the predictor
    is closing the gap with real representations over training.
    """
    _plot_loss_components(
        axes,
        {
            'Real-repr loss': _get_metric(log, 'real_scoring_loss'),
            'Pred-repr loss': _get_metric(log, 'pred_scoring_loss'),
        },
        'Scoring Loss',
        normalize=False,
    )


def plot_prediction_loss(axes: plt.Axes, log: TrainingLog) -> None:
    """Training-only metric."""
    _plot_loss(axes, _get_metric(log, 'prediction_loss'), 'Prediction Loss')


def plot_score_consistency_loss(axes: plt.Axes, log: TrainingLog) -> None:
    """MSE between predicted-repr scores and real-repr scores (functional alignment)."""
    _plot_combined_loss(
        axes,
        _get_metric(log, 'score_consistency_loss'),
        _get_metric(log, 'val_score_consistency_loss'),
        'Score Consistency Loss',
    )


def plot_repr_dist_kl_loss(axes: plt.Axes, log: TrainingLog) -> None:
    """Symmetric KL between pred and real representation distributions."""
    _plot_combined_loss(
        axes,
        _get_metric(log, 'repr_dist_kl_loss'),
        _get_metric(log, 'val_repr_dist_kl_loss'),
        'Representation Distribution Similarity KL Loss',
        mark_best_val=False,
    )


def plot_component_losses_weighted(axes: plt.Axes, log: TrainingLog) -> None:
    """Training component losses scaled by their actual loss weights.

    Shows the true gradient contribution of each component (i.e. what the optimizer
    actually sees), which is the most direct view for diagnosing weight imbalances.
    Weights are read from the training log config; defaults match the model constructor.
    """
    pw = float(log.config.get('prediction_loss_weight', 1.0))
    plw = float(log.config.get('predictability_loss_weight', 0.2))
    kw = float(log.config.get('repr_kl_loss_weight', 0.01))
    scw = float(log.config.get('score_consistency_loss_weight', 0.1))
    dkw = float(log.config.get('repr_dist_kl_loss_weight', 0.01))

    _plot_loss_components(
        axes,
        {
            'Scoring (×1)': _get_metric(log, 'scoring_loss'),
            f'Prediction (×{pw:g})': _weighted_metric(_get_metric(log, 'prediction_loss'), pw),
            f'Predictability (×{plw:g})': _weighted_metric(_get_metric(log, 'predictability_loss'), plw),
            f'Repr KL (×{kw:g})': _weighted_metric(_get_metric(log, 'repr_kl_loss'), kw),
            f'Score Consistency (×{scw:g})': _weighted_metric(_get_metric(log, 'score_consistency_loss'), scw),
            f'Dist KL (×{dkw:g})': _weighted_metric(_get_metric(log, 'repr_dist_kl_loss'), dkw),
        },
        'Weighted Component Losses',
        normalize=False,
    )


def plot_diagnostic_grad_norms(axes: plt.Axes, log: TrainingLog) -> None:
    """Gradient norms for encoder, predictor, and scorer (log scale, normalized)."""
    _plot_loss_components(
        axes,
        {
            'Encoder': _get_metric(log, 'encoder_grad_norm'),
            'Predictor': _get_metric(log, 'predictor_grad_norm'),
            'Scorer': _get_metric(log, 'scorer_grad_norm'),
        },
        'Gradient Norms by Subnetwork',
        normalize=True,
    )


def plot_predictor_input_proj_norms(axes: plt.Axes, log: TrainingLog) -> None:
    """L2 norms of predictor input projections (log scale)."""
    _plot_loss_components(
        axes,
        {
            'Prompt proj.': _get_metric(log, 'predictor_prompt_proj_norm'),
            'Features proj.': _get_metric(log, 'predictor_feat_proj_norm'),
            'Model proj.': _get_metric(log, 'predictor_model_proj_norm'),
        },
        'Predictor Input Projection Norms',
        normalize=False,
    )


def plot_pred_real_repr_diagnostics(axes: plt.Axes, log: TrainingLog) -> None:
    """Norms and mean per-dim variance for predicted vs encoded response representations."""
    _plot_loss_components(
        axes,
        {
            'Predicted repr norm': _get_metric(log, 'pred_repr_norm'),
            'Real repr norm': _get_metric(log, 'real_repr_norm'),
            'Predicted repr variance': _get_metric(log, 'pred_repr_variance'),
            'Real repr variance': _get_metric(log, 'real_repr_variance'),
        },
        'Predicted / Real Representation Norms & Variance',
        normalize=False,
    )


def plot_score_variance_diagnostics(axes: plt.Axes, log: TrainingLog) -> None:
    """Batch score variance and model vs prompt decomposition (real-repr scores)."""
    _plot_score_variance_decomposition(
        axes,
        _get_metric(log, 'score_total_variance'),
        _get_metric(log, 'score_model_variance_ratio'),
        _get_metric(log, 'score_prompt_variance_ratio'),
    )


def plot_grad_attr_embeddings(axes: plt.Axes, log: TrainingLog) -> None:
    """Mean gradient×|input| attribution for prompt and model embeddings."""
    _plot_loss_components(
        axes,
        {
            'Prompt embedding': _get_metric(log, 'grad_attr_prompt_embedding'),
            'Model embedding': _get_metric(log, 'grad_attr_model_embedding'),
        },
        'Gradient attribution',
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
    scw = float(log.config.get('score_consistency_loss_weight', 0.1))
    dkw = float(log.config.get('repr_dist_kl_loss_weight', 0.01))

    _plot_loss_components(
        axes,
        {
            'Scoring (×1)': _get_metric(log, 'scoring_loss'),
            f'Prediction (×{pw:g})': _weighted_metric(_get_metric(log, 'prediction_loss'), pw),
            f'Predictability (×{plw:g})': _weighted_metric(_get_metric(log, 'predictability_loss'), plw),
            f'Representation KL (×{kw:g})': _weighted_metric(_get_metric(log, 'repr_kl_loss'), kw),
            f'Score Consistency (×{scw:g})': _weighted_metric(_get_metric(log, 'score_consistency_loss'), scw),
            f'Distribution Similarity KL (×{dkw:g})': _weighted_metric(_get_metric(log, 'repr_dist_kl_loss'), dkw),
        },
        'Weighted Component Losses Normalized',
        normalize=True,
    )
