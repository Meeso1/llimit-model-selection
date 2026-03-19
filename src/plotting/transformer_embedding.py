import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting.core import (
    plot_combined_loss as _plot_combined_loss,
    plot_combined_accuracy as _plot_combined_accuracy,
    plot_loss_components as _plot_loss_components,
    plot_positive_metric as _plot_positive_metric,
    _get_metric,
)


def plot_metrics(log: TrainingLog) -> plt.Figure:
    """Create a figure with all metrics for the transformer embedding model.

    Layout: 3 rows × 2 columns.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    plot_loss(axes[0, 0], log)
    plot_accuracy(axes[0, 1], log)
    plot_modality_norms(axes[1, 0], log)
    plot_gradient_norms(axes[1, 1], log)
    plot_interaction_norm(axes[2, 0], log)
    axes[2, 1].set_visible(False)

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
    )
