"""Store Sales panel loading and 16-day backtests used by notebooks 03 and 04.

Models accept ``(history, future_X)`` and return one ``pred`` per
``(store_nbr, family, date)``. Historical sales stay available for fitting;
future targets and sales-derived flags are withheld during evaluation.
"""

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd


K = ["store_nbr", "family"]
H = 16
FOLDS = {
    "F1": pd.Timestamp("2017-06-12"),
    "F2": pd.Timestamp("2017-06-28"),
    "F3": pd.Timestamp("2017-07-14"),
}
HOLDOUT = {"HOLDOUT": pd.Timestamp("2017-07-30")}
TEST_CUTOFF = pd.Timestamp("2017-08-15")
PROCESSED_PATH = (
    Path(__file__).resolve().parents[2]
    / "kaggle_comp" / "data" / "processed.parquet"
)
TARGET_COLUMNS = ["sales", "log_sales", "active", "zero_series"]


def load_panel(path: str | Path | None = None) -> pd.DataFrame:
    """Read notebook 02's Parquet output, independently of the current directory."""
    path = Path(path) if path is not None else PROCESSED_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed data not found: {path}. "
            "Run kaggle_comp/notebooks/02_preprocessing.ipynb first."
        )
    try:
        panel = pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "Reading processed.parquet requires pyarrow. "
            "Install the project dependencies or run: python -m pip install pyarrow"
        ) from exc

    required = set(K + ["date", "sales", "id", "split"])
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Processed data is missing columns: {sorted(missing)}")
    panel["date"] = pd.to_datetime(panel["date"])
    if panel[K + ["date"]].isna().any().any():
        raise ValueError("Processed data contains missing store, family, or date keys")
    if panel.duplicated(K + ["date"]).any():
        raise ValueError("Processed data contains duplicate store-family-date rows")
    return panel.sort_values(K + ["date"]).reset_index(drop=True)


def rmsle(y_true, y_pred) -> float:
    """RMSLE on sales, clipping negative forecasts to zero."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actuals and predictions must have the same nonempty shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actuals and predictions must be finite")
    if (actual < 0).any():
        raise ValueError("Actual sales cannot be negative")
    error = np.log1p(np.maximum(predicted, 0)) - np.log1p(actual)
    return float(np.sqrt(np.mean(error ** 2)))


def zero_series(history: pd.DataFrame) -> pd.MultiIndex:
    """Find pairs with zero total sales in the last 30 days of THIS history."""
    cutoff = history["date"].max()
    recent = history.loc[history["date"] > cutoff - pd.Timedelta(days=30)]
    totals = recent.groupby(K, observed=True)["sales"].sum(min_count=1)
    return totals.index[totals.eq(0)]


def random_walk(history: pd.DataFrame, future_X: pd.DataFrame) -> pd.DataFrame:
    """Repeat the last observed sales value, including an observed zero."""
    last = (
        history.sort_values("date")
        .groupby(K, observed=True)["sales"].last().rename("pred")
    )
    out = future_X[K + ["date"]].merge(
        last.reset_index(), on=K, how="left", validate="many_to_one"
    )
    # A pair with no history has no last observation; use the zero fallback.
    out["pred"] = out["pred"].fillna(0)
    return out


def finalize(predictions: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Apply the history-only zero-sales rule and nonnegative sales constraint."""
    out = predictions[K + ["date", "pred"]].copy()
    if out.duplicated(K + ["date"]).any():
        raise ValueError("Model returned duplicate store-family-date predictions")
    out["pred"] = out["pred"].astype(float)
    is_zero = pd.MultiIndex.from_frame(out[K]).isin(zero_series(history))
    out.loc[is_zero, "pred"] = 0.0
    if not np.isfinite(out["pred"]).all():
        raise ValueError("Model returned missing or non-finite predictions")
    out["pred"] = out["pred"].clip(lower=0)
    return out


def run_fold(
    model_fn: Callable, panel: pd.DataFrame, cutoff: str | pd.Timestamp
) -> pd.DataFrame:
    """Forecast the 16 dates after cutoff and attach held-out actuals for scoring."""
    cutoff = pd.Timestamp(cutoff)
    history = panel.loc[panel["date"] <= cutoff].copy()
    # Notebook 02's zero_series flag uses sales through August 15; it is not
    # valid at an earlier backtest cutoff. Recompute it through zero_series().
    history = history.drop(columns=["zero_series"], errors="ignore")
    future = panel.loc[
        (panel["date"] > cutoff)
        & (panel["date"] <= cutoff + pd.Timedelta(days=H))
    ].copy()
    if history.empty:
        raise ValueError("No training observations are available before the cutoff")
    if not np.isfinite(history["sales"].to_numpy(dtype=float)).all():
        raise ValueError("Training sales must be known and finite")

    expected_dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=H)
    actual_dates = pd.DatetimeIndex(future["date"].unique()).sort_values()
    counts = future.groupby(K, observed=True).size()
    if (
        not actual_dates.equals(expected_dates)
        or not counts.eq(H).all()
        or future.duplicated(K + ["date"]).any()
    ):
        raise ValueError("Each forecast series must have 16 distinct consecutive dates")
    if not np.isfinite(future["sales"].to_numpy(dtype=float)).all():
        raise ValueError("Backtesting requires known sales for all 16 forecast dates")

    future_X = future.drop(columns=TARGET_COLUMNS, errors="ignore")
    predictions = finalize(model_fn(history, future_X), history)
    out = future[K + ["date", "sales"]].merge(
        predictions, on=K + ["date"], how="left", validate="one_to_one"
    )
    if len(predictions) != len(out) or out["pred"].isna().any():
        raise ValueError("Model must return one prediction for every forecast row")
    return out


def backtest(
    model_fn: Callable,
    panel: pd.DataFrame,
    folds: Mapping[str, pd.Timestamp],
    name: str = "model",
) -> dict:
    """Report each fold's RMSLE and their unweighted mean."""
    if not folds:
        raise ValueError("At least one backtest fold is required")
    scores = {}
    for fold, cutoff in folds.items():
        out = run_fold(model_fn, panel, cutoff)
        scores[fold] = rmsle(out["sales"], out["pred"])
        print(
            f"{name:<28} {fold:<8} cutoff {pd.Timestamp(cutoff):%Y-%m-%d}  "
            f"RMSLE {scores[fold]:.4f}"
        )
    return {"scores": scores, "mean": float(np.mean(list(scores.values())))}
