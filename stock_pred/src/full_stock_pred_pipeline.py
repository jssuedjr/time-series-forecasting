"""Reusable one-step-ahead univariate stock forecasting experiment.

Assumes shared_code.backtest.backtest.univariate_backtest returns canonical
columns: origin_date, target_date, horizon, model, y_true, y_pred, error.
"""
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from shared_code.evaluation.metrics import evaluate_forecasts, mse, rmse, mae, rmsle, interval_summary
from shared_code.evaluation.diagnostics import forecast_error_diagnostics

from shared_code.backtest.backtest import univariate_backtest
from shared_code.evaluation.predictive_distribution import EmpiricalPredictiveDistribution


@dataclass
class ForecastResults:
    model_name: str
    train: pd.Series
    validation: pd.DataFrame
    test: pd.DataFrame
    calibration_errors: np.ndarray
    validation_diagnostics: pd.DataFrame
    evaluation: pd.DataFrame
    forecast: pd.DataFrame
    predictive_samples: np.ndarray
    distribution: EmpiricalPredictiveDistribution


def run_forecasting_pipeline(
    prices: pd.Series,
    model_class,
    horizon: int = 1,
    window_size: int = 252,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    interval_level: float = 0.95,
    model_kwargs: dict | None = None,
    train_end: int | None = None,
    validation_end: int | None = None,
) -> ForecastResults:
    """Backtest, calibrate on validation, evaluate on test, forecast next session.

    Optional integer boundaries freeze the original train/validation/test split
    when new observations are appended. For a clean held-out test, pass only the
    original historical dataset; production data can be added later for fitting.
    """
    if horizon != 1:
        raise NotImplementedError("This pipeline currently supports horizon=1 only")
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price index must be unique and chronological")
    prices = prices.astype(float)
    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Prices must be finite")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Invalid split fractions")
    if not (0 < interval_level < 1):
        raise ValueError("interval_level must be between 0 and 1")
    if not isinstance(window_size, int) or window_size < 2:
        raise ValueError("window_size must be an integer >= 2")

    n = len(prices)
    train_end = int(n * train_fraction) if train_end is None else train_end
    validation_end = (int(n * (train_fraction + validation_fraction))
                      if validation_end is None else validation_end)
    if not (window_size <= train_end < validation_end < n):
        raise ValueError("Need window_size <= train_end < validation_end < n")
    if validation_end - train_end < 3 or n - validation_end < 1:
        raise ValueError("Insufficient validation or test observations")

    kwargs = dict(model_kwargs or {})
    model_factory = partial(model_class, **kwargs)
    name = getattr(model_class, "__name__", type(model_class).__name__)
    options = dict(model_class=model_factory, initial_train_size=window_size,
                   horizon=1, strategy="rolling", window_size=window_size)

    validation = univariate_backtest(
        prices, **options, start_origin=train_end, end_origin=validation_end
    ).copy()
    test = univariate_backtest(
        prices, **options, start_origin=validation_end, end_origin=n
    ).copy()
    required = {"origin_date", "target_date", "y_true", "y_pred", "error"}
    for frame in (validation, test):
        if not required.issubset(frame.columns):
            raise ValueError("Update backtest.py to use the canonical columns: "
                             + ", ".join(sorted(required)))
        frame["model"] = name
    validation["split"] = "validation"
    test["split"] = "test"
    if not set(validation["target_date"]).isdisjoint(test["target_date"]):
        raise AssertionError("Validation and test targets overlap")

    calibration_errors = validation["error"].to_numpy(dtype=float).copy()
    # Reuse exactly the same inverse-ECDF quantile convention as the final forecast.
    error_distribution = EmpiricalPredictiveDistribution(0.0, calibration_errors)
    q_low, q_high = error_distribution.interval(interval_level)
    test["predictive_mean"] = test["y_pred"] + error_distribution.mean()
    test["interval_lower"] = test["y_pred"] + q_low
    test["interval_upper"] = test["y_pred"] + q_high
    test["covered"] = (
        (test["y_true"] >= test["interval_lower"])
        & (test["y_true"] <= test["interval_upper"])
    )

    evaluation = evaluate_forecasts(
        pd.concat([validation, test], ignore_index=True),
        metrics=('mse', 'rmse', 'mae', 'rmsle'), group_by='split'
    ).rename(columns={'split': 'period'})
    mask = evaluation['period'] == 'test'
    evaluation.loc[mask, 'predictive_mean_MSE'] = mse(test['y_true'], test['predictive_mean'])
    evaluation.loc[mask, 'predictive_mean_RMSE'] = rmse(test['y_true'], test['predictive_mean'])
    evaluation.loc[mask, 'predictive_mean_MAE'] = mae(test['y_true'], test['predictive_mean'])
    evaluation.loc[mask, 'predictive_mean_RMSLE'] = rmsle(test['y_true'], test['predictive_mean'])
    summary = interval_summary(test, interval_level).iloc[0]
    evaluation.loc[mask, 'interval_coverage'] = summary['empirical_test_coverage']
    evaluation.loc[mask, 'mean_interval_width'] = summary['mean_interval_width_dollars']

    # Fit with the same rolling-window policy used for historical forecasts.
    model = model_factory()
    model.fit(prices.iloc[-window_size:].to_numpy())
    prediction = np.asarray(model.predict(1), dtype=float)
    if prediction.shape != (1,) or not np.isfinite(prediction).all():
        raise ValueError("Model must return one finite forecast for horizon=1")
    point = float(prediction[0])
    distribution = EmpiricalPredictiveDistribution(point, calibration_errors)
    lower, upper = distribution.interval(interval_level)
    forecast = pd.DataFrame([{
        "origin_date": prices.index[-1],
        "last_close": float(prices.iloc[-1]),
        "point_forecast": point,
        "predictive_mean": distribution.mean(),
        "predictive_median": distribution.quantile(0.5),
        "predictive_variance": distribution.variance(),
        "interval_lower": lower,
        "interval_upper": upper,
        "interval_level": interval_level,
        "n_calibration_errors": len(calibration_errors),
    }])
    return ForecastResults(
        model_name=name, train=prices.iloc[:train_end],
        validation=validation, test=test,
        calibration_errors=calibration_errors,
        validation_diagnostics=forecast_error_diagnostics(calibration_errors),
        evaluation=evaluation, forecast=forecast,
        predictive_samples=distribution.samples,
        distribution=distribution,
    )
