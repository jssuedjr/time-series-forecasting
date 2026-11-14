import numpy as np
import pandas as pd

from shared_code.validation import validate_numeric_pair


def rmse(y_true, y_pred) -> float:
    a, p = validate_numeric_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((a - p) ** 2)))

def rmsle(y_true, y_pred) -> float:
    """Root mean squared log error, log1p convention (Kaggle)."""
    a, p = validate_numeric_pair(y_true, y_pred, nonnegative=True)
    return float(np.sqrt(np.mean((np.log1p(a) - np.log1p(p)) ** 2)))

def mase(y_true, y_pred) -> float:
    ...

def mae(y_true, y_pred) -> float:
    a, p = validate_numeric_pair(y_true, y_pred)
    return float(np.mean(np.abs(a - p)))

def evaluate_forecasts(
    results: pd.DataFrame,
    metrics: tuple[str, ...] = ("rmse", "mae", "rmsle"),
    group_by: str | list[str] | None = None
) -> pd.DataFrame:
    """
    Compute metrics from a results table with columns
    `actual` and `prediction`, optionally per group
    (e.g. group_by=["model", "horizon"]).
    """
    functions = {"rmse": rmse, "mae": mae, "rmsle": rmsle}
    unknown = set(metrics) - set(functions)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")

    def score(part):
        return pd.Series({
            "n": len(part),
            **{m: functions[m](part["actual"], part["prediction"])
               for m in metrics},
        })

    if group_by is None:
        return score(results).to_frame().T

    return results.groupby(group_by).apply(score).reset_index()
