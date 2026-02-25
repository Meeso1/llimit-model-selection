"""Metrics computation for length prediction models."""

import numpy as np
from src.preprocessing.simple_scaler import SimpleScaler


def compute_length_prediction_metrics(
    log_predictions: np.ndarray,  # [n_samples] - scaled(log(raw_length))
    log_actuals: np.ndarray,  # [n_samples] - scaled(log(raw_length))
    scaler: SimpleScaler,
) -> dict[str, float]:
    """
    Compute evaluation metrics for length prediction.
    
    Metrics computed:
    - avg_relative_error: abs(1 - predicted/actual) (lower is better, min 0.0)
    - avg_relative_ratio: mean(predicted/actual) (should be close to 1.0)
    - stddev_ratio: stddev(predictions)/stddev(actuals) (should be close to 1.0)
    - rmse: root mean squared error in scaled space
    - mae: mean absolute error in descaled space
    
    Args:
        log_predictions: Predicted lengths (scaled(log(raw_length)))  # [n_samples]
        log_actuals: Actual lengths (scaled(log(raw_length)))  # [n_samples]
        scaler: Scaler used for standardization
        
    Returns:
        Dictionary of metrics
    """    
    log_predictions_descaled = scaler.inverse_transform(log_predictions)  # [n_samples]
    log_actuals_descaled = scaler.inverse_transform(log_actuals)  # [n_samples]
    
    predictions_descaled = np.exp(log_predictions_descaled)  # [n_samples]
    actuals_descaled = np.exp(log_actuals_descaled)  # [n_samples]
    
    # Avoid division by zero
    actuals_nonzero = np.where(actuals_descaled == 0, 1e-6, actuals_descaled)
    
    # Average relative error magnitude: abs(1 - predicted/actual)
    relative_ratios = predictions_descaled / actuals_nonzero
    avg_relative_error = float(np.mean(np.abs(1 - relative_ratios)))
    
    # Average relative ratio: mean(predicted/actual)
    avg_relative_ratio = float(np.mean(relative_ratios))
    
    # Stddev ratio: stddev(predictions)/stddev(actuals)
    pred_std = np.std(predictions_descaled)
    actual_std = np.std(actuals_descaled)
    stddev_ratio = float(pred_std / actual_std) if actual_std > 0 else 1.0
    
    # RMSE in scaled space
    rmse = float(np.sqrt(np.mean((log_predictions - log_actuals) ** 2)))
    
    # Mean absolute error in descaled space
    mae = float(np.mean(np.abs(predictions_descaled - actuals_descaled)))
    
    # Synthetic accuracy metric - 1 if avg_relative_error is 0, 0.5 if avg_relative_error is 1, halves every time avg_relative_error increases by 1
    accuracy = 1 / 2**np.minimum(avg_relative_error, 10)
    
    return {
        "accuracy": accuracy,
        "avg_relative_error": avg_relative_error,
        "avg_relative_ratio": avg_relative_ratio,
        "stddev_ratio": stddev_ratio,
        "rmse": rmse,
        "mae": mae,
    }
