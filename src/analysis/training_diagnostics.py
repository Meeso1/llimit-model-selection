import math
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn as nn


def split_tensor_with_grad(
    tensor: torch.Tensor,   # [batch_size, n_features], requires_grad=True, grad populated
    feature_names: list[str],
) -> dict[str, torch.Tensor]:  # name -> [batch_size, 1], detached leaf with .grad set
    """Split a tensor and its gradient into individually named leaf tensors.

    Call after ``loss.backward()`` when ``tensor.requires_grad=True``.  Each
    returned tensor is a detached leaf whose ``.grad`` is set to the
    corresponding column of ``tensor.grad``, ready for
    ``EpochDiagnosticsAccumulator.update_gradient_attribution``.

    Returns an empty dict if ``tensor.grad`` is ``None``.

    Args:
        tensor: Tensor that participated in the forward pass with
            ``requires_grad=True``; its ``.grad`` must be populated.
        feature_names: One name per column; must match ``tensor.shape[1]``.
    """
    if tensor.grad is None:
        return {}
    if tensor.shape[1] != len(feature_names):
        raise ValueError(
            f"tensor has {tensor.shape[1]} columns but {len(feature_names)} names were given"
        )
    result = {}
    for i, name in enumerate(feature_names):
        leaf = tensor[:, i : i + 1].detach()   # [batch_size, 1], leaf, no grad tracking
        leaf.grad = tensor.grad[:, i : i + 1].clone()
        result[name] = leaf
    return result


class EpochDiagnosticsAccumulator:
    """Accumulates per-batch training diagnostics over one epoch.

    Instantiate at epoch start, call update_* methods per batch, call
    to_dict() at epoch end.
    """

    def __init__(self) -> None:
        self._grad_norm_sums: dict[str, float] = defaultdict(float)
        self._n_grad_norm_batches: int = 0

        self._repr_norm_sums: dict[str, float] = defaultdict(float)
        self._repr_var_sums: dict[str, float] = defaultdict(float)
        self._n_repr_batches: int = 0

        self._score_total_var_sum: float = 0.0
        self._score_model_ratio_sum: float = 0.0
        self._score_prompt_ratio_sum: float = 0.0
        self._n_score_batches: int = 0

        self._attr_sums: dict[str, torch.Tensor] = {}   # name -> accumulated [dim]
        self._attr_batches: dict[str, int] = defaultdict(int)  # name -> batch count

    def update_grad_norms(
        self,
        param_groups: dict[str, Iterable[nn.Parameter]],
    ) -> None:
        """Accumulate gradient norms per named parameter group.

        Call after ``loss.backward()``, before ``clip_grad_norm_``.
        Parameters with ``grad is None`` are skipped.

        Args:
            param_groups: Mapping of group name to its parameters.
        """
        for name, params in param_groups.items():
            total_norm_sq = 0.0
            for p in params:
                if p.grad is not None:
                    total_norm_sq += p.grad.data.norm(2).item() ** 2
            self._grad_norm_sums[name] += math.sqrt(total_norm_sq)
        self._n_grad_norm_batches += 1

    def update_representation_stats(
        self,
        representations: dict[str, torch.Tensor],  # name -> [batch_size, dim]
    ) -> None:
        """Accumulate norm and variance statistics for intermediate representations.

        For each tensor, records the mean L2 norm across the batch and the mean
        per-dimension variance across the batch.  Low variance signals
        representation collapse.

        Args:
            representations: Mapping of name to intermediate tensor.
        """
        for name, repr_tensor in representations.items():
            self._repr_norm_sums[name] += repr_tensor.norm(dim=1).mean().item()
            self._repr_var_sums[name] += repr_tensor.var(dim=0).mean().item()
        self._n_repr_batches += 1

    def update_score_variance(
        self,
        scores: torch.Tensor,     # [batch_size]
        model_ids: torch.Tensor,  # [batch_size]
    ) -> None:
        """Decompose batch score variance into model-driven and prompt-driven portions.

        Computes per-model score means, then splits total variance into
        between-model variance (model effect) and within-model variance (prompt
        effect).  Skipped gracefully when fewer than 2 distinct models appear
        in the batch.

        Args:
            scores: Per-sample scores (detached).
            model_ids: Integer model ID for each score.
        """
        if model_ids.unique().numel() < 2:
            return

        total_var = scores.var().item()
        if total_var < 1e-10:
            return

        max_id = int(model_ids.max().item())
        model_means = torch.zeros(max_id + 1, dtype=scores.dtype, device=scores.device)
        model_counts = torch.zeros(max_id + 1, dtype=scores.dtype, device=scores.device)
        model_means.scatter_add_(0, model_ids, scores)
        model_counts.scatter_add_(0, model_ids, torch.ones_like(scores))
        mask = model_counts > 0
        model_means[mask] = model_means[mask] / model_counts[mask]

        present_means = model_means[mask]
        model_var = present_means.var().item() if present_means.numel() > 1 else 0.0

        residuals = scores - model_means[model_ids]
        prompt_var = residuals.var().item()

        self._score_total_var_sum += total_var
        self._score_model_ratio_sum += model_var / total_var
        self._score_prompt_ratio_sum += prompt_var / total_var
        self._n_score_batches += 1

    def update_gradient_attribution(
        self,
        inputs: dict[str, torch.Tensor],  # name -> [batch_size, dim], grad must be populated
    ) -> None:
        """Accumulate gradient × input attribution for named input tensors.

        Call after ``loss.backward()``.  For each tensor, its ``.grad`` must be
        populated: leaf tensors need ``requires_grad_(True)`` set before the
        forward pass; non-leaf tensors need ``retain_grad()`` called before the
        forward pass.  Entries whose ``.grad`` is ``None`` are silently skipped.

        For dim-1 tensors (e.g. individual feature slices from
        ``split_features_for_grad_attr``), attribution is a scalar emitted
        directly as ``grad_attr_{name}``.  For higher-dimensional tensors
        (e.g. whole embeddings), attribution is averaged across dimensions into
        a single ``grad_attr_{name}`` scalar representing mean per-dimension
        importance.

        Args:
            inputs: Mapping of name to input tensor whose gradient attribution
                should be tracked.
        """
        for name, tensor in inputs.items():
            if tensor.grad is None:
                continue
            with torch.no_grad():
                attr = (tensor.grad.abs() * tensor.abs()).mean(dim=0)  # [dim]
            if name not in self._attr_sums:
                self._attr_sums[name] = attr.cpu().clone()
            else:
                self._attr_sums[name] += attr.cpu()
            self._attr_batches[name] += 1

    def to_dict(self) -> dict[str, float]:
        """Return all accumulated metrics averaged over their respective batch counts.

        Only groups that had at least one ``update_*`` call contribute keys.

        Returns:
            Flat dict of metric name to averaged float value.  Intended for use
            as ``TrainingHistoryEntry.additional_metrics``.
        """
        result: dict[str, float] = {}

        if self._n_grad_norm_batches > 0:
            scale = 1.0 / self._n_grad_norm_batches
            for name, total in self._grad_norm_sums.items():
                result[f"{name}_grad_norm"] = total * scale

        if self._n_repr_batches > 0:
            scale = 1.0 / self._n_repr_batches
            for name in self._repr_norm_sums:
                result[f"{name}_norm"] = self._repr_norm_sums[name] * scale
                result[f"{name}_variance"] = self._repr_var_sums[name] * scale

        if self._n_score_batches > 0:
            scale = 1.0 / self._n_score_batches
            result["score_total_variance"] = self._score_total_var_sum * scale
            result["score_model_variance_ratio"] = self._score_model_ratio_sum * scale
            result["score_prompt_variance_ratio"] = self._score_prompt_ratio_sum * scale

        for name, total in self._attr_sums.items():
            n = self._attr_batches[name]
            avg_attr = (total / n).tolist()
            if isinstance(avg_attr, float):
                # tolist() on a 0-d tensor returns a float
                result[f"grad_attr_{name}"] = avg_attr
            elif len(avg_attr) == 1:
                result[f"grad_attr_{name}"] = avg_attr[0]
            else:
                result[f"grad_attr_{name}"] = sum(avg_attr) / len(avg_attr)

        return result
