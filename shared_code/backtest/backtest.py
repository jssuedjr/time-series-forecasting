from collections.abc import Callable

import numpy as np
import pandas as pd


def _positive_int(name, value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")


def _get_window_start(t, strategy, window_size):
    if strategy == "expanding":
        return 0

    return max(0, t - window_size)


def univariate_backtest(
    data: pd.Series,
    model_class: Callable,
    initial_train_size: int,
    horizon: int = 1,
    step: int = 1,
    strategy: str = "expanding",
    window_size: int | None = None,
    start_origin: int | None = None,
    end_origin: int | None = None
) -> pd.DataFrame:
    """
    Rolling-origin backtesting for univariate models.

    start_origin:
        First forecast target position (inclusive).

    end_origin:
        Evaluation target boundary (exclusive).
        All forecast targets must be before this position.
    """

    if not isinstance(data, pd.Series):
        raise TypeError("Expected a pandas Series")

    if data.empty:
        raise ValueError("Data cannot be empty")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Data must be chronologically sorted")

    if not data.index.is_unique:
        raise ValueError("Data index must be unique")

    values = data.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Data must contain finite values")

    for name, value in {
        "initial_train_size": initial_train_size,
        "horizon": horizon,
        "step": step
    }.items():
        _positive_int(name, value)

    n = len(values)

    if initial_train_size + horizon > n:
        raise ValueError("Not enough observations")

    if strategy not in {"expanding", "rolling", "capped"}:
        raise ValueError("Invalid window strategy")

    if strategy in {"rolling", "capped"}:
        _positive_int("window_size", window_size)

    if strategy == "rolling":
        if initial_train_size != window_size:
            raise ValueError(
                "Rolling requires initial_train_size == window_size"
            )

    if strategy == "capped":
        if initial_train_size > window_size:
            raise ValueError(
                "Capped requires initial_train_size <= window_size"
            )

    if start_origin is None:
        start_origin = initial_train_size

    if end_origin is None:
        end_origin = n

    _positive_int("start_origin", start_origin)
    _positive_int("end_origin", end_origin)

    if not (
        initial_train_size <= start_origin
        and start_origin + horizon <= end_origin <= n
    ):
        raise ValueError("Invalid evaluation boundaries")

    results = []

    # Last valid origin must have all targets inside
    # the evaluation boundary.
    last_origin_exclusive = end_origin - horizon + 1

    for t in range(
        start_origin,
        last_origin_exclusive,
        step
    ):
        start = _get_window_start(
            t, strategy, window_size
        )

        train = values[start:t]
        test = values[t:t + horizon]

        model = model_class()
        model.fit(train)

        forecast = np.asarray(
            model.predict(horizon),
            dtype=float
        )

        if forecast.shape != (horizon,):
            raise ValueError(
                f"Expected forecast shape ({horizon},), "
                f"received {forecast.shape}"
            )

        if not np.isfinite(forecast).all():
            raise ValueError("Non-finite forecast values")

        for h in range(horizon):
            results.append({
                "origin": data.index[t - 1],
                "target": data.index[t + h],
                "horizon": h + 1,
                "strategy": strategy,
                "train_start": data.index[start],
                "train_size": len(train),
                "actual": test[h],
                "predicted": forecast[h],
                "error": test[h] - forecast[h]
            })

    return pd.DataFrame(results)


#################################################################
# Minimal rolling-origin backtest with the shared result schema
#################################################################

def rolling_origin_backtest(
    data: pd.Series,
    model_class: Callable,
    model_name: str,
    horizon: int,
    series_id: str = "series",
    origin_start=None,
    origin_end=None,
    window: str = "expanding",
    window_size: int | None = None,
    min_train_size: int = 30,
    interval_level: float | None = None
) -> pd.DataFrame:
    """
    Rolling-origin backtest returning the shared columns:

        model, series_id, forecast_origin, target_date, horizon,
        actual, prediction [, lower, upper]

    data:
        Date-indexed Series. Horizon h means h rows after the
        origin (trading days for stocks, calendar days for sales).

    origin_start / origin_end:
        Date range (inclusive) of forecast origins. The origin is
        the last date in the training data. Origins without all
        h targets in the data are dropped.

    window:
        "expanding" uses all rows up to the origin,
        "fixed" uses the last `window_size` rows.

    interval_level:
        If set and the model has predict_interval(horizon, level),
        the lower and upper bounds are recorded.
    """

    if not isinstance(data, pd.Series):
        raise TypeError("Expected a pandas Series")

    if not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise ValueError("Data index must be sorted and unique")

    if window not in {"expanding", "fixed"}:
        raise ValueError("window must be 'expanding' or 'fixed'")

    _positive_int("horizon", horizon)

    if window == "fixed":
        _positive_int("window_size", window_size)
        min_train_size = max(min_train_size, window_size)

    values = data.to_numpy(dtype=float)
    index = data.index

    mask = np.arange(len(data)) <= len(data) - 1 - horizon
    if origin_start is None:
        # default: earliest origin with enough training data
        mask &= np.arange(len(data)) >= min_train_size - 1
    else:
        mask &= index >= pd.Timestamp(origin_start)
    if origin_end is not None:
        mask &= index <= pd.Timestamp(origin_end)

    positions = np.flatnonzero(mask)

    if len(positions) == 0:
        raise ValueError("No valid forecast origins")

    if positions[0] + 1 < min_train_size:
        raise ValueError("First origin has too little training data")

    results = []

    for pos in positions:
        end = pos + 1
        start = 0 if window == "expanding" else end - window_size

        model = model_class()
        model.fit(values[start:end])
        forecast = np.asarray(model.predict(horizon), dtype=float)

        bounds = None
        if interval_level is not None and hasattr(model, "predict_interval"):
            bounds = model.predict_interval(horizon, interval_level)

        for h in range(1, horizon + 1):
            row = {
                "model": model_name,
                "series_id": series_id,
                "forecast_origin": index[pos],
                "target_date": index[pos + h],
                "horizon": h,
                "actual": values[pos + h],
                "prediction": forecast[h - 1]
            }
            if bounds is not None:
                row["lower"] = bounds[0][h - 1]
                row["upper"] = bounds[1][h - 1]
            results.append(row)

    return pd.DataFrame(results)
