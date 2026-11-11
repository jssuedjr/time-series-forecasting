
"""
Shared input validation utilities.

These functions check structural requirements for
evaluation, visualization, and diagnostics.

They do not test statistical model assumptions.
"""

import numpy as np
import pandas as pd

def validate_numeric_pair(y_true, y_pred, nonnegative=False):
    """
    Validate two arrays used for forecast evaluation.

    Parameters
    ----------
    y_true, y_pred : array-like
        Observed and predicted values, already aligned.
    nonnegative : bool
        Require values >= 0 (e.g. RMSLE).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated numeric arrays.
    """
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(actual) == 0:
        raise ValueError("Inputs cannot be empty.")

    if len(actual) != len(predicted):
        raise ValueError("Input lengths must match.")

    try:
        actual = actual.astype(float)
        predicted = predicted.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Inputs must be numeric.") from exc

    if not np.isfinite(actual).all():
        raise ValueError("y_true contains non-finite values.")

    if not np.isfinite(predicted).all():
        raise ValueError("y_pred contains non-finite values.")

    if nonnegative:
        if (actual < 0).any() or (predicted < 0).any():
            raise ValueError("Inputs must be nonnegative.")

    return actual, predicted


def validate_series(series, min_length=3):
    """
    Validate a univariate numeric sequence.

    The caller is responsible for ensuring
    chronological ordering and appropriate spacing.
    """
    values = np.asarray(series)

    if values.ndim != 1:
        raise ValueError("Series must be one-dimensional.")

    if len(values) < min_length:
        raise ValueError(
            f"Series requires at least {min_length} observations."
        )

    try:
        values = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Series must be numeric.") from exc

    if not np.isfinite(values).all():
        raise ValueError("Series contains non-finite values.")

    return values


def validate_lags(lags, n_obs):
    """Check that an integer lag count is valid."""
    if isinstance(lags, (bool, np.bool_)) or not isinstance(
        lags, (int, np.integer)
    ):
        raise ValueError("Lags must be an integer.")

    if not 1 <= lags < n_obs:
        raise ValueError(
            "Lags must be positive and less than sample size."
        )

    return int(lags)


def validate_forecast_results(
    results: pd.DataFrame,
    required_columns=("y_true", "y_pred"),
    group_by=None
):
    """
    Validate the structure of forecast results.

    Optional columns may include:
        model, series_id, origin_date,
        target_date, horizon
    """
    if not isinstance(results, pd.DataFrame):
        raise TypeError("Results must be a DataFrame.")

    if results.empty:
        raise ValueError("Results cannot be empty.")

    missing = set(required_columns) - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    groups = (
        [group_by] if isinstance(group_by, str)
        else list(group_by) if group_by is not None
        else []
    )

    missing_groups = set(groups) - set(results.columns)

    if missing_groups:
        raise ValueError(
            f"Missing grouping columns: {sorted(missing_groups)}"
        )

    if results[groups].isna().any().any():
        raise ValueError("Grouping columns contain missing values.")

    # Check the forecast values without changing
    # the caller's DataFrame.
    validate_numeric_pair(
        results["y_true"].to_numpy(),
        results["y_pred"].to_numpy()
    )

    return results.copy()


def validate_backtest_results(results: pd.DataFrame):
    """Validate the shared backtesting output schema."""
    required = (
        "origin_date",
        "target_date",
        "horizon",
        "y_true",
        "y_pred",
        "model"
    )

    validated = validate_forecast_results(
        results,
        required_columns=required
    )

    origin = pd.to_datetime(
        validated["origin_date"], errors="raise"
    )
    target = pd.to_datetime(
        validated["target_date"], errors="raise"
    )

    if origin.isna().any() or target.isna().any():
        raise ValueError("Forecast dates cannot be missing.")

    if not (target > origin).all():
        raise ValueError(
            "Target dates must follow forecast origins."
        )

    horizons = pd.to_numeric(
        validated["horizon"], errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(horizons).all():
        raise ValueError("Horizons must be finite.")

    if not ((horizons >= 1) & (horizons % 1 == 0)).all():
        raise ValueError(
            "Horizons must be positive integers."
        )

    if validated["model"].isna().any():
        raise ValueError("Model identifiers cannot be missing.")

    keys = ["model", "origin_date", "target_date"]

    if "series_id" in validated.columns:
        keys.insert(0, "series_id")
        if validated["series_id"].isna().any():
            raise ValueError("Series identifiers cannot be missing.")

    if validated.duplicated(subset=keys).any():
        raise ValueError("Duplicate forecast records detected.")

    return validated
