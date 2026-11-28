"""
Shared data and forecast-result formats (see shared_code/README.md).

Observed data (long format)
    series_id | date | y | <extra columns ...>

Forecast / backtest results (produced by backtest.univariate_backtest and
checked by validation.validate_backtest_results)
    origin_date | target_date | horizon | model | y_true | y_pred
    [| series_id | error | interval_lower | interval_upper | ...]

`origin_date` is the last date whose information is available when the
forecast is made (e.g. the close of that trading day). Errors use the
convention y_true - y_pred.
"""

from collections.abc import Sequence

import pandas as pd

ID_COL = "series_id"
DATE_COL = "date"
TARGET_COL = "y"

FORECAST_COLUMNS = (
    "origin_date",
    "target_date",
    "horizon",
    "model",
    "y_true",
    "y_pred",
)
INTERVAL_COLUMNS = ("interval_lower", "interval_upper")


def to_long(
    df: pd.DataFrame,
    series_id: str,
    target_col: str,
    date_col: str | None = None,
    extra_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Convert a single-series frame to the shared long format.

    If `date_col` is None the index is used as the date.
    """
    frame = df.reset_index() if date_col is None else df.copy()
    date_source = frame.columns[0] if date_col is None else date_col

    out = pd.DataFrame({
        ID_COL: series_id,
        DATE_COL: pd.to_datetime(frame[date_source]),
        TARGET_COL: frame[target_col].astype(float),
    })

    for col in extra_cols:
        out[col] = frame[col].to_numpy()

    return out.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)
