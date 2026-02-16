"""Helper utilities for computing statistics on model scores."""

import numpy as np
from typing import Any


def compute_model_scores_stats(scores: dict[str, float]) -> dict[str, Any]:
    """
    Compute summary statistics for model scores.
    
    Args:
        scores: Dictionary mapping model names to their scores
        
    Returns:
        Dictionary with statistics: mean, median, std, min, max, top10%, top90%,
        best_model_name, worst_model_name
    """
    if not scores:
        return {}
    
    model_names = list(scores.keys())
    score_values = np.array([scores[name] for name in model_names])
    
    # Find best and worst models
    best_idx = np.argmax(score_values)
    worst_idx = np.argmin(score_values)
    
    # Compute percentiles
    top_10_pct = float(np.percentile(score_values, 90))
    top_90_pct = float(np.percentile(score_values, 10))
    
    return {
        "mean": float(np.mean(score_values)),
        "median": float(np.median(score_values)),
        "std": float(np.std(score_values)),
        "min": float(np.min(score_values)),
        "max": float(np.max(score_values)),
        "top10%": top_10_pct,
        "top90%": top_90_pct,
        "best_model_name": model_names[best_idx],
        "worst_model_name": model_names[worst_idx],
    }
