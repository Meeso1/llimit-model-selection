"""Metrics computation for length prediction models."""

import numpy as np
from src.preprocessing.simple_scaler import SimpleScaler


def compute_length_prediction_metrics(
    predictions: np.ndarray,  # [n_samples] - scaled
    actuals: np.ndarray,  # [n_samples] - scaled
    scaler: SimpleScaler,
) -> dict[str, float]:
    """
    Compute evaluation metrics for length prediction.
    
    Metrics computed:
    - avg_relative_error: 1 - abs(1 - predicted/actual) (higher is better, max 1.0)
    - avg_relative_ratio: mean(predicted/actual) (should be close to 1.0)
    - stddev_ratio: stddev(predictions)/stddev(actuals) (should be close to 1.0)
    - rmse: root mean squared error in scaled space
    - mae: mean absolute error in descaled space
    
    Args:
        predictions: Predicted lengths (scaled)  # [n_samples]
        actuals: Actual lengths (scaled)  # [n_samples]
        scaler: Scaler used for standardization
        
    Returns:
        Dictionary of metrics
    """
    predictions_descaled = scaler.inverse_transform(predictions)  # [n_samples]
    actuals_descaled = scaler.inverse_transform(actuals)  # [n_samples]
    
    # Avoid division by zero
    actuals_nonzero = np.where(actuals_descaled == 0, 1e-6, actuals_descaled)
    
    # Average relative error magnitude: 1 - abs(1 - predicted/actual)
    relative_ratios = predictions_descaled / actuals_nonzero
    avg_relative_error = float(1 - np.mean(np.abs(1 - relative_ratios)))
    
    # Average relative ratio: mean(predicted/actual)
    avg_relative_ratio = float(np.mean(relative_ratios))
    
    # Stddev ratio: stddev(predictions)/stddev(actuals)
    pred_std = np.std(predictions_descaled)
    actual_std = np.std(actuals_descaled)
    stddev_ratio = float(pred_std / actual_std) if actual_std > 0 else 1.0
    
    # RMSE in scaled space
    rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
    
    # Mean absolute error in descaled space
    mae = float(np.mean(np.abs(predictions_descaled - actuals_descaled)))
    
    return {
        "avg_relative_error": avg_relative_error,
        "avg_relative_ratio": avg_relative_ratio,
        "stddev_ratio": stddev_ratio,
        "rmse": rmse,
        "mae": mae,
    }
