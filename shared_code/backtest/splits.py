"""Reusable chronological train/validation/test splitting."""
import pandas as pd


def chronological_split(data: pd.Series | pd.DataFrame,
                        train_fraction: float = 0.60,
                        val_fraction: float = 0.20):
    """Original fraction-based split (kept for backward compatibility)."""
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("Expected pandas Series or DataFrame")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1
            and train_fraction + val_fraction < 1):
        raise ValueError("Invalid split fractions")
    if not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise ValueError("Index must be chronological and unique")
    n = len(data)
    a = int(n * train_fraction)
    b = int(n * (train_fraction + val_fraction))
    if not 0 < a < b < n:
        raise ValueError("Not enough data for three splits")
    return data.iloc[:a], data.iloc[a:b], data.iloc[b:]


def fixed_date_split(data: pd.Series | pd.DataFrame,
                     train_end_date, validation_end_date, test_end_date):
    """Return frozen train/validation/test partitions (cutoffs inclusive).

    Observations newer than test_end_date are excluded from the benchmark.
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("Expected pandas Series or DataFrame")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex of trading dates")
    if not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise ValueError("Index must be chronological and unique")
    a, b, c = map(pd.Timestamp, (train_end_date, validation_end_date, test_end_date))
    if not a < b < c:
        raise ValueError("Require train_end_date < validation_end_date < test_end_date")
    train = data.loc[data.index <= a].copy()
    validation = data.loc[(data.index > a) & (data.index <= b)].copy()
    test = data.loc[(data.index > b) & (data.index <= c)].copy()
    if min(len(train), len(validation), len(test)) < 1:
        raise ValueError("Each partition must have at least one observation")
    return train, validation, test
