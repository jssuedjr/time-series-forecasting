import pandas as pd

def rmse(y_true, y_pred) -> float:
    ...

def rmsle(y_true, y_pred) -> float:
    ...

def mase(y_true, y_pred) -> float:
    ...

def mae(y_true, y_pred) -> float:
    ...

def evaluate_forecasts(
    results: pd.DataFrame,
    metrics: tuple[str, ...] = ("rmse", "mae", "rmsle"),
    group_by: str | list[str] | None = None
) -> pd.DataFrame:
    ...