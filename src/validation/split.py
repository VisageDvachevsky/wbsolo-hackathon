"""
Time-based validation splits that mimic the test structure.
"""
import pandas as pd


def time_holdout_split(df: pd.DataFrame, n_test_timestamps: int = 8):
    """
    Hold out the last n_test_timestamps per route as validation.

    Returns (train_df, val_df) where val_df has exactly the same
    structure as the competition test set (n_test_timestamps per route).
    """
    timestamps = sorted(df["timestamp"].unique())
    cutoff = timestamps[-n_test_timestamps]
    train_part = df[df["timestamp"] < cutoff].copy()
    val_part = df[df["timestamp"] >= cutoff].copy()
    return train_part, val_part


def expanding_window_splits(df: pd.DataFrame, n_folds: int = 3, n_test_timestamps: int = 8):
    """
    Generate multiple time-forward validation folds.

    Each fold holds out n_test_timestamps from a different day,
    using everything before as training.
    """
    timestamps = sorted(df["timestamp"].unique())
    total = len(timestamps)
    # Space folds evenly across the last portion of data
    step = n_test_timestamps * 2  # gap between fold starts
    folds = []
    for i in range(n_folds):
        end_idx = total - i * step
        start_idx = end_idx - n_test_timestamps
        if start_idx < total // 2:
            break
        val_timestamps = timestamps[start_idx:end_idx]
        cutoff = val_timestamps[0]
        train_part = df[df["timestamp"] < cutoff].copy()
        val_part = df[df["timestamp"].isin(val_timestamps)].copy()
        folds.append((train_part, val_part))
    return folds
