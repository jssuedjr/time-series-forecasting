"""Bella's fixed-predictor SARIMAX model from notebook 04."""

import numpy as np
import pandas as pd

from shared_code.backtest import kaggle as bt


FIXED_PREDICTORS = ("log_promo", "holiday", "event", "closed")


def _model_frame(rows):
    """Use the column names and numeric types expected by StatsForecast."""
    frame = rows[bt.K + ["date"]].copy()
    frame["unique_id"] = rows["store_nbr"].astype(str) + "_" + rows["family"].astype(str)
    frame["ds"] = rows["date"]
    frame["log_promo"] = np.log1p(rows["onpromotion"].astype(float))
    for name in ("holiday", "event", "closed"):
        frame[name] = rows[name].astype(float)
    return frame


def fixed_forecast(history, future_X, train_days=365, n_jobs=4):
    """Forecast the next 16 days. Set train_days to None to fit all history."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    keys = bt.K + ["date"]
    cutoff = history["date"].max()
    zero_pairs = bt.zero_series(history)
    training = history
    if train_days is not None:
        training = history.loc[history["date"] > cutoff - pd.Timedelta(days=train_days)]
    training = training.loc[~pd.MultiIndex.from_frame(training[bt.K]).isin(zero_pairs)]
    zero_future = pd.MultiIndex.from_frame(future_X[bt.K]).isin(zero_pairs)
    result = future_X[keys].copy()
    if zero_future.all():
        return result.assign(pred=0.0)

    train = _model_frame(training)
    train["y"] = np.log1p(training["sales"].to_numpy(dtype=float))
    future = _model_frame(future_X.loc[~zero_future])

    # A coefficient can only be estimated if its predictor varies during training.
    varies = train.groupby("unique_id")[list(FIXED_PREDICTORS)].std().gt(0)
    groups = {}
    for series_id, varies_in_series in varies.iterrows():
        columns = tuple(name for name in FIXED_PREDICTORS if varies_in_series[name])
        groups.setdefault(columns, []).append(series_id)

    predictions = []
    for columns, series_ids in groups.items():
        columns = list(columns)
        training_rows = train.loc[
            train["unique_id"].isin(series_ids), ["unique_id", "ds", "y"] + columns
        ]
        future_rows = future.loc[
            future["unique_id"].isin(series_ids), ["unique_id", "ds"] + columns
        ]
        model = StatsForecast(
            models=[AutoARIMA(season_length=7, ic="bic")], freq="D", n_jobs=n_jobs
        )
        predictions.append(
            model.forecast(df=training_rows, h=bt.H, X_df=future_rows if columns else None)
        )

    forecast = pd.concat(predictions, ignore_index=True)
    forecast["pred"] = np.expm1(forecast["AutoARIMA"])
    lookup = future[["unique_id", "ds"] + keys]
    by_key = lookup.merge(
        forecast[["unique_id", "ds", "pred"]], on=["unique_id", "ds"],
        how="left", validate="one_to_one"
    )
    result = result.merge(by_key[keys + ["pred"]], on=keys, how="left", validate="one_to_one")
    zero_future = pd.MultiIndex.from_frame(result[bt.K]).isin(zero_pairs)
    result.loc[zero_future, "pred"] = 0.0
    if not np.isfinite(result["pred"]).all():
        raise ValueError("Missing or invalid forecast for a store and family.")
    return result
