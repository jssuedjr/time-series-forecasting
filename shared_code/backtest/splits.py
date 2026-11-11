
import pandas as pd
import numpy as np

def chronological_split(
    data: pd.Series | pd.DataFrame,
    train_fraction: float = 0.60,
    val_fraction: float = 0.20
):
    """Split time-ordered data into train, validation, and test."""

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("Expected pandas Series or DataFrame")

    if not 0 < train_fraction < 1:
        raise ValueError("Invalid training fraction")

    if not 0 < val_fraction < 1:
        raise ValueError("Invalid validation fraction")

    if train_fraction + val_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to < 1")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Data must be chronologically sorted")

    n = len(data)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))

    if not (0 < train_end < val_end < n):
        raise ValueError("Not enough data for all three splits")

    train = data.iloc[:train_end]
    validation = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]

    return train, validation, test
