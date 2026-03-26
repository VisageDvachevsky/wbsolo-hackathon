"""
Post-processing: bias correction, clipping.
"""
import numpy as np


def clip_predictions(y_pred: np.ndarray, min_val: float = 0) -> np.ndarray:
    """Clip predictions to non-negative values."""
    return np.maximum(y_pred, min_val)


def correct_bias(y_pred: np.ndarray, y_true_sum: float) -> np.ndarray:
    """
    Multiplicatively adjust predictions so their sum matches y_true_sum.
    This zeroes out the Relative Bias component of the metric.
    """
    pred_sum = y_pred.sum()
    if pred_sum > 0:
        return y_pred * (y_true_sum / pred_sum)
    return y_pred
