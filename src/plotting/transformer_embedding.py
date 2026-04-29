import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting.core import (
    plot_combined_loss as _plot_combined_loss,
    plot_combined_accuracy as _plot_combined_accuracy,
    plot_loss_components as _plot_loss_components,
    plot_positive_metric as _plot_positive_metric,
    plot_score_variance_decomposition as _plot_score_variance_decomposition,
    _get_metric,
)


def plot_metrics(log: TrainingLog) -> plt.Figure:
    """Create a figure with all per-epoch metrics for TransformerEmbeddingModel.

    Includes training-diagnostics from ``EpochDiagnosticsAccumulator``.  Post-training
    metrics (e.g. sensitivity ablations in ``final_metrics``) are not plotted.

    Layout: 4 rows × 2 columns.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))

    plot_loss(axes[0, 0], log)
    plot_accuracy(axes[0, 1], log)
    plot_modality_norms(axes[1, 0], log)
    plot_gradient_norms(axes[1, 1], log)
    plot_interaction_norm(axes[2, 0], log)
    plot_score_variance(axes[2, 1], log)
    plot_modality_variances(axes[3, 0], log)
    plot_grad_attr_embeddings(axes[3, 1], log)

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


def plot_modality_norms(axes: plt.Axes, log: TrainingLog) -> None:
    """Plot per-modality projected L2 norms (log scale, no normalization).

    Shows relative magnitudes of prompt_emb, feat, and model_proj contributions.
    """
    _plot_loss_components(
        axes,
        {
            'Prompt emb proj': _get_metric(log, 'prompt_emb_proj_norm'),
            'Feat proj': _get_metric(log, 'feat_proj_norm'),
            'Model proj': _get_metric(log, 'model_proj_norm'),
        },
        'Modality Projected Norms (Training)',
        normalize=False,
    )


def plot_modality_variances(axes: plt.Axes, log: TrainingLog) -> None:
    """Mean per-dimension variance of projected tensors (collapse diagnostic)."""
    _plot_loss_components(
        axes,
        {
            'Prompt emb proj': _get_metric(log, 'prompt_emb_proj_variance'),
            'Feat proj': _get_metric(log, 'feat_proj_variance'),
            'Model proj': _get_metric(log, 'model_proj_variance'),
            'Interaction': _get_metric(log, 'interaction_variance'),
        },
        'Modality / Interaction Variance (Training)',
        normalize=False,
    )


def plot_gradient_norms(axes: plt.Axes, log: TrainingLog) -> None:
    """Plot gradient norms per component (log scale, normalized to start at 1).

    Shows relative learning rates across transformer, projection, and scoring head.
    """
    _plot_loss_components(
        axes,
        {
            'Transformer': _get_metric(log, 'transformer_grad_norm'),
            'Projection': _get_metric(log, 'projection_grad_norm'),
            'Scoring head': _get_metric(log, 'scoring_head_grad_norm'),
        },
        'Gradient Norms by Component (Training)',
        normalize=True,
    )


def plot_interaction_norm(axes: plt.Axes, log: TrainingLog) -> None:
    """Plot L2 norm of interaction vector (log scale).

    Collapse toward zero indicates prompt and model representations becoming orthogonal.
    """
    _plot_positive_metric(
        axes,
        _get_metric(log, 'interaction_norm'),
        'Interaction Norm (Training)',
        ylabel='log(norm)',
        show_min_line=False,
    )


def plot_score_variance(axes: plt.Axes, log: TrainingLog) -> None:
    _plot_score_variance_decomposition(
        axes,
        _get_metric(log, 'score_total_variance'),
        _get_metric(log, 'score_model_variance_ratio'),
        _get_metric(log, 'score_prompt_variance_ratio'),
    )


def plot_grad_attr_embeddings(axes: plt.Axes, log: TrainingLog) -> None:
    """Mean gradient×|input| attribution for prompt embedding and model embedding."""
    _plot_loss_components(
        axes,
        {
            'Prompt embedding': _get_metric(log, 'grad_attr_prompt_embedding'),
            'Model embedding': _get_metric(log, 'grad_attr_model_embedding'),
        },
        'Gradient attribution — embeddings (Training)',
        normalize=False,
    )
