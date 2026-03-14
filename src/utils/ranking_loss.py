"""Shared pairwise ranking loss for prompt-sensitive scoring models."""

from typing import Literal
import torch
import torch.nn.functional as F


PairwiseRankingLossType = Literal["margin_ranking", "bradley_terry"]


def compute_pairwise_ranking_loss(
    loss_type: PairwiseRankingLossType,
    scores_a: torch.Tensor,  # [batch_size]
    scores_b: torch.Tensor,  # [batch_size]
    labels: torch.Tensor,  # [batch_size], 1.0 if a wins, -1.0 if b wins
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Compute pairwise ranking loss (stateless).

    Args:
        loss_type: "margin_ranking" (hinge) or "bradley_terry" (sigmoid cross-entropy).
        scores_a: Scores for model A.
        scores_b: Scores for model B.
        labels: 1.0 if A wins, -1.0 if B wins.
        margin: Margin for margin_ranking (ignored for bradley_terry).

    Returns:
        Scalar or per-sample loss tensor, depending on reduction.
    """
    if loss_type == "margin_ranking":
        return F.margin_ranking_loss(scores_a, scores_b, labels, margin=margin)
    if loss_type == "bradley_terry":
        diff = scores_a - scores_b  # [batch_size]
        targets = (labels + 1.0) / 2.0  # {-1, 1} -> {0, 1}
        return F.binary_cross_entropy_with_logits(diff, targets)
    raise ValueError(f"Unknown loss_type: {loss_type}")
