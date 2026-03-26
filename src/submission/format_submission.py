"""
Format predictions into the competition submission CSV.
"""
import pandas as pd
import numpy as np


def create_submission(test_df: pd.DataFrame, predictions: np.ndarray, output_path: str):
    """
    Create a submission CSV matching the competition format.

    Parameters
    ----------
    test_df : DataFrame with 'id' column
    predictions : array of predicted target_1h values
    output_path : path to save the CSV
    """
    sub = pd.DataFrame({
        "id": test_df["id"].values,
        "y_pred": predictions,
    })
    sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}, shape: {sub.shape}")
    return sub
