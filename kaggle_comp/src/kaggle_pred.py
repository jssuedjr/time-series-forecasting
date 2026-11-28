"""Run a Store Sales forecast using a fixed or selected predictor list."""

from functools import partial
from importlib.metadata import version
from pathlib import Path
import platform

import pandas as pd

from shared_code.backtest import kaggle as bt
from .bella import FIXED_PREDICTORS, fixed_forecast
from .selection import forecast_selected, select_predictors

MODES = {"Fixed predictors": "fixed", "Stepwise selection": "stepwise"}
KEYS = bt.K + ["date"]
BELLA_INPUTS = KEYS + ["sales", "onpromotion", "holiday", "event", "closed"]


def prepare_run(results_dir, train_days, choice):
    """Make a results folder for this training window and predictor choice."""
    if choice not in MODES:
        raise ValueError("Choose 'Fixed predictors' or 'Stepwise selection'.")
    if train_days not in (365, None):
        raise ValueError("Use 365 for the recent window or None for all available dates.")
    window = "recent_365_days" if train_days == 365 else "all_history"
    out = Path(results_dir) / window / MODES[choice]
    out.mkdir(parents=True, exist_ok=True)
    packages = ["numpy", "pandas", "pyarrow", "scipy", "statsforecast", "joblib"]
    versions = {name: version(name) for name in packages}
    versions["python"] = platform.python_version()
    pd.Series(versions, name="version").to_csv(out / "package_versions.csv", index_label="package")
    return out


def choose_predictors(panel, train_days, choice, out, series_per_group=2, workers=4,
                      max_steps=4, min_improvement=0.0005):
    """Return the predictor label, columns and development score for this run."""
    if choice == "Fixed predictors":
        return " + ".join(FIXED_PREDICTORS), list(FIXED_PREDICTORS), None
    if choice != "Stepwise selection":
        raise ValueError("Choose 'Fixed predictors' or 'Stepwise selection'.")
    label, columns, sample, details, ranking = select_predictors(
        panel, train_days, series_per_group=series_per_group, n_jobs=workers,
        max_steps=max_steps, min_improvement=min_improvement
    )
    sample.to_csv(out / "selection_series.csv", index=False)
    details.to_csv(out / "selection_details.csv", index=False)
    ranking.to_csv(out / "selection_scores.csv")
    path = ranking.dropna(subset=["selection_step"]).sort_values("selection_step")
    path.to_csv(out / "selection_path.csv")
    return label, columns, float(ranking.loc[label, "selection_rmsle"])


def forecast_period(panel, cutoff, train_days, choice, columns, workers=4):
    """Forecast the next 16 days for every store and family at this cutoff."""
    cutoff = pd.Timestamp(cutoff)
    if choice == "Stepwise selection":
        return forecast_selected(panel, cutoff, columns, train_days, n_jobs=workers)
    if choice != "Fixed predictors" or list(columns) != list(FIXED_PREDICTORS):
        raise ValueError("The fixed model requires Bella's four predictors.")

    end = cutoff + pd.Timedelta(days=bt.H)
    future = panel.loc[panel.date.gt(cutoff) & panel.date.le(end)]
    mask = panel.date.le(end)
    if train_days is not None:
        mask &= panel.date.gt(cutoff - pd.Timedelta(days=max(30, train_days)))
    model_data = panel.loc[mask, BELLA_INPUTS]
    model = partial(fixed_forecast, train_days=train_days, n_jobs=workers)
    if cutoff == bt.TEST_CUTOFF:
        if not future.is_test.all() or future.empty:
            raise ValueError("Expected Kaggle test rows after the final training date.")
        history = model_data.loc[model_data.date.le(cutoff)]
        future_x = future[[c for c in BELLA_INPUTS if c != "sales"]]
        predictions = bt.finalize(model(history, future_x), history)
    else:
        predictions = bt.run_fold(model, model_data, cutoff)

    forecasts = future[KEYS + ["id"]].merge(predictions, on=KEYS, validate="one_to_one", how="left")
    if forecasts.pred.isna().any() or len(forecasts) != len(predictions):
        raise ValueError("Expected one forecast for each store, family and date.")
    return forecasts, None
