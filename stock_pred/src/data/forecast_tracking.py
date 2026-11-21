"""Persist issued forecasts and reconcile them with completed daily closes.

Place this file at stock_pred/src/data/forecast_tracking.py.
"""
from pathlib import Path
import pandas as pd

HISTORY_PATH = Path(__file__).resolve().parents[2] / "data" / "forecast_history.csv"
KEY = ["ticker", "model", "origin_date", "target_date"]
FORECAST_FIELDS = KEY + [
    "point_forecast", "predictive_mean", "interval_lower",
    "interval_upper", "interval_level",
]


def save_forecast(record):
    """Append an issued forecast, preserving the first record on reruns."""
    missing = [field for field in FORECAST_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Missing forecast fields: {missing}")

    lower = float(record["interval_lower"])
    upper = float(record["interval_upper"])
    level = float(record["interval_level"])
    if lower > upper or not 0 < level < 1:
        raise ValueError("Invalid prediction interval bounds or level")

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([record])

    if HISTORY_PATH.exists():
        history = pd.read_csv(HISTORY_PATH)
        history = pd.concat([history, new], ignore_index=True)
    else:
        history = new

    # Preserve the forecast that was actually issued first.
    history = history.drop_duplicates(subset=KEY, keep="first")
    history.to_csv(HISTORY_PATH, index=False)
    return history


def reconcile_actuals(history, market_data):
    """Fill observed closes and score forecasts without changing issued bounds.

    market_data must contain only completed sessions with Date and Close columns.
    """
    history = history.copy()
    actuals = market_data[["Date", "Close"]].copy()
    actuals = actuals.rename(columns={"Date": "target_date", "Close": "new_actual"})

    history["target_date"] = pd.to_datetime(history["target_date"]).dt.normalize()
    actuals["target_date"] = pd.to_datetime(actuals["target_date"]).dt.normalize()
    actuals = actuals.drop_duplicates(subset=["target_date"], keep="last")

    # Reconciliation may be run repeatedly; preserve known actuals if this
    # market-data batch doesn't contain their dates.
    if "actual" not in history.columns:
        history["actual"] = float("nan")

    history = history.merge(actuals, on="target_date", how="left", validate="many_to_one")
    history["actual"] = history["new_actual"].combine_first(history["actual"])
    history = history.drop(columns=["new_actual"])

    history["point_error"] = history["actual"] - history["point_forecast"]
    history["predictive_mean_error"] = history["actual"] - history["predictive_mean"]

    covered = (
        (history["actual"] >= history["interval_lower"])
        & (history["actual"] <= history["interval_upper"])
    )
    history["covered"] = covered.astype("boolean").mask(history["actual"].isna(), pd.NA)

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(HISTORY_PATH, index=False)
    return history
