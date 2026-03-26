"""
WAPE + |Relative Bias| metric, matching the competition implementation.
"""
import numpy as np


class WapePlusRbias:
    """Calculates WAPE + |Relative Bias|."""

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the competition metric."""
        wape = (np.abs(y_pred - y_true)).sum() / y_true.sum()
        rbias = np.abs(y_pred.sum() / y_true.sum() - 1)
        return wape + rbias

    def calculate_components(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Return WAPE and Relative Bias separately for diagnostics."""
        wape = (np.abs(y_pred - y_true)).sum() / y_true.sum()
        rbias_signed = y_pred.sum() / y_true.sum() - 1
        return {
            "wape": wape,
            "rbias": np.abs(rbias_signed),
            "rbias_signed": rbias_signed,
            "total": wape + np.abs(rbias_signed),
        }
