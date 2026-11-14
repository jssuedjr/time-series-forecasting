"""
Shared data and forecast-result formats.

Observed data (long format)
    series_id | date | target | <extra columns ...>

Forecast / backtest results
    model | series_id | forecast_origin | target_date | horizon
    | actual | prediction [| lower | upper]

`forecast_origin` is the last date whose information is available
when the forecast is made (e.g. the close of that trading day).
"""

from collections.abc import Sequence

import pandas as pd

ID_COL = "series_id"
DATE_COL = "date"
TARGET_COL = "target"

FORECAST_COLUMNS = (
    "model",
    "series_id",
    "forecast_origin",
    "target_date",
    "horizon",
    "actual",
    "prediction",
)
INTERVAL_COLUMNS = ("lower", "upper")


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
