import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting.core import (
    plot_combined_loss as _plot_combined_loss,
    plot_combined_accuracy as _plot_combined_accuracy,
    plot_loss_components as _plot_loss_components,
    plot_score_variance_decomposition as _plot_score_variance_decomposition,
    _get_metric,
)


def plot_metrics(log: TrainingLog) -> plt.Figure:
    """Create a figure with all per-epoch metrics for DnEmbeddingModel.

    Includes training-diagnostics (norms, gradient norms, score decomposition,
    embedding-level gradient attribution).  Post-training-only metrics (e.g.
    sensitivity ablations) live in ``final_metrics`` and are not plotted here.

    Layout: 4 rows × 2 columns.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))

    plot_loss(axes[0, 0], log)
    plot_accuracy(axes[0, 1], log)
    plot_modality_norms(axes[1, 0], log)
    plot_modality_variances(axes[1, 1], log)
    plot_gradient_norms(axes[2, 0], log)
    plot_score_variance(axes[2, 1], log)
    plot_grad_attr_embeddings(axes[3, 0], log)
    axes[3, 1].set_visible(False)

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
    """Per-modality projected L2 norms after input projections (log scale)."""
    _plot_loss_components(
        axes,
        {
            'Prompt emb. proj.': _get_metric(log, 'prompt_emb_proj_norm'),
            'Prompt feat. proj.': _get_metric(log, 'prompt_feat_proj_norm'),
            'Model emb. proj.': _get_metric(log, 'model_emb_proj_norm'),
        },
        'Input Projection Norms',
        normalize=False,
    )


def plot_modality_variances(axes: plt.Axes, log: TrainingLog) -> None:
    """Mean per-dimension variance of projected representations (collapse diagnostic)."""
    _plot_loss_components(
        axes,
        {
            'Prompt emb. proj.': _get_metric(log, 'prompt_emb_proj_variance'),
            'Prompt feat. proj.': _get_metric(log, 'prompt_feat_proj_variance'),
            'Model emb. proj.': _get_metric(log, 'model_emb_proj_variance'),
        },
        'Input Projection Variance',
        normalize=False,
    )


def plot_gradient_norms(axes: plt.Axes, log: TrainingLog) -> None:
    """Gradient norms: trunk vs each input projection (log scale, normalized)."""
    _plot_loss_components(
        axes,
        {
            'Trunk': _get_metric(log, 'trunk_grad_norm'),
            'Prompt emb. proj.': _get_metric(log, 'prompt_emb_proj_grad_norm'),
            'Prompt feat. proj.': _get_metric(log, 'prompt_feat_proj_grad_norm'),
            'Model emb. proj.': _get_metric(log, 'model_emb_proj_grad_norm'),
        },
        'Gradient Norms by Component',
        normalize=True,
    )


def plot_score_variance(axes: plt.Axes, log: TrainingLog) -> None:
    """Score variance decomposition: log(total variance) and model/prompt ratios."""
    _plot_score_variance_decomposition(
        axes,
        _get_metric(log, 'score_total_variance'),
        _get_metric(log, 'score_model_variance_ratio'),
        _get_metric(log, 'score_prompt_variance_ratio'),
    )


def plot_grad_attr_embeddings(axes: plt.Axes, log: TrainingLog) -> None:
    """Mean gradient×|input| attribution for prompt and model embeddings (log scale)."""
    _plot_loss_components(
        axes,
        {
            'Prompt embedding': _get_metric(log, 'grad_attr_prompt_embedding'),
            'Model embedding': _get_metric(log, 'grad_attr_model_embedding'),
        },
        'Gradient attribution',
        normalize=False,
    )

