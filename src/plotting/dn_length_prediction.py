import matplotlib.pyplot as plt

from src.utils.training_logger import TrainingLog
from src.plotting.core import (
    plot_loss_components as _plot_loss_components,
    _get_metric,
)
from src.plotting._length_prediction_shared import (
    plot_loss,
    plot_accuracy,
    plot_mae,
    plot_rmse,
    plot_relative_error,
    plot_relative_ratio,
    plot_stddev_ratio,
)
from src.plotting.dn_embedding import (
    plot_modality_norms,
    plot_modality_variances,
    plot_grad_attr_embeddings,
)


def plot_metrics(log: TrainingLog, skip_first_n_epochs: int = 0) -> plt.Figure:
    """Create a figure with all per-epoch metrics for DnEmbeddingLengthPredictionModel.

    Includes training diagnostics (projection norms/variances, gradient norms).
    Gradient attribution (``plot_grad_attr_embeddings``) is available as a
    standalone function but not included here.

    When skip_first_n_epochs > 0, the y-axis range of metric plots is computed
    from that epoch onward (all epochs are still drawn).

    Layout: 5 rows × 2 columns (10 panels).
    """
    fig, axes = plt.subplots(5, 2, figsize=(14, 25))

    plot_loss(axes[0, 0], log)
    plot_accuracy(axes[0, 1], log)
    plot_mae(axes[1, 0], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_rmse(axes[1, 1], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_relative_error(axes[2, 0], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_relative_ratio(axes[2, 1], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_stddev_ratio(axes[3, 0], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_modality_norms(axes[3, 1], log)
    plot_modality_variances(axes[4, 0], log)
    plot_gradient_norms(axes[4, 1], log)

    fig.tight_layout()
    return fig


def plot_metrics_4by2(log: TrainingLog, skip_first_n_epochs: int = 0) -> plt.Figure:
    """Create a figure with selected metrics for DnEmbeddingLengthPredictionModel.

    Layout: 4 rows × 2 columns.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    
    plot_loss(axes[0, 0], log)
    plot_accuracy(axes[0, 1], log)
    plot_mae(axes[1, 0], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_relative_error(axes[1, 1], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_relative_ratio(axes[2, 0], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_stddev_ratio(axes[2, 1], log, skip_first_n_epochs=skip_first_n_epochs)
    plot_modality_norms(axes[3, 0], log)
    plot_gradient_norms(axes[3, 1], log)
    
    fig.tight_layout()
    return fig


def plot_gradient_norms(axes: plt.Axes, log: TrainingLog) -> None:
    """Gradient norms: trunk, input projections, and model-id embedding (log scale, normalized)."""
    _plot_loss_components(
        axes,
        {
            'Trunk': _get_metric(log, 'trunk_grad_norm'),
            'Prompt emb. proj.': _get_metric(log, 'prompt_emb_proj_grad_norm'),
            'Prompt feat. proj.': _get_metric(log, 'prompt_feat_proj_grad_norm'),
            'Model emb. proj.': _get_metric(log, 'model_emb_proj_grad_norm'),
            'Model ID emb.': _get_metric(log, 'model_id_embedding_grad_norm'),
        },
        'Gradient Norms by Component',
        normalize=True,
    )
