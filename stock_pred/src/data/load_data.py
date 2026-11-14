"""
Download and load daily prices from Yahoo Finance (yfinance).

Raw files: <raw_dir>/<TICKER>.csv plus <raw_dir>/metadata.json with
the fetch time, request parameters and a checksum, so every
experiment can state exactly which data snapshot it used.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
# A bar for "today" is only treated as final after this NY time.
FINAL_BAR_TIME = (16, 30)

RAW_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close",
               "Volume", "Dividends", "Stock Splits"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None,
    raw_dir: Path,
) -> dict:
    """
    Download unadjusted-for-dividends daily bars and save them.

    auto_adjust=False keeps both `Close` (split-adjusted quoted
    close) and `Adj Close` (also dividend-adjusted); actions=True
    keeps the dividend and split events for the adjustment check.
    """
    import yfinance as yf

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    fetched_at = datetime.now(timezone.utc)
    now_ny = fetched_at.astimezone(NY)
    today_ny = pd.Timestamp(now_ny.date())
    bar_final = (now_ny.hour, now_ny.minute) >= FINAL_BAR_TIME

    meta = {
        "source": "Yahoo Finance via yfinance",
        "yfinance_version": yf.__version__,
        "fetched_at_utc": fetched_at.isoformat(),
        "fetched_at_new_york": now_ny.isoformat(),
        "request": {"start": start, "end": end, "interval": "1d",
                    "auto_adjust": False, "actions": True},
        "files": {},
    }

    for ticker in tickers:
        df = yf.Ticker(ticker).history(
            start=start, end=end, interval="1d",
            auto_adjust=False, actions=True,
        )
        if df.empty:
            raise RuntimeError(f"No data returned for {ticker}")

        df.index = pd.DatetimeIndex(df.index.tz_localize(None).normalize(),
                                    name="date")
        df = df[[c for c in RAW_COLUMNS if c in df.columns]]

        dropped_partial = None
        if not bar_final and df.index[-1] == today_ny:
            dropped_partial = str(today_ny.date())
            df = df.iloc[:-1]

        path = raw_dir / f"{ticker}.csv"
        df.to_csv(path)

        meta["files"][ticker] = {
            "path": path.name,
            "rows": int(len(df)),
            "first_date": str(df.index[0].date()),
            "last_date": str(df.index[-1].date()),
            "sha256": _sha256(path),
            "dropped_incomplete_bar": dropped_partial,
        }

    (raw_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_raw(ticker: str, raw_dir: Path) -> pd.DataFrame:
    """Read one saved raw file as a date-indexed frame."""
    path = Path(raw_dir) / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run with --refresh to download"
        )
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def load_metadata(raw_dir: Path) -> dict:
    meta = json.loads((Path(raw_dir) / "metadata.json").read_text())
    for ticker, info in meta["files"].items():
        current = _sha256(Path(raw_dir) / info["path"])
        info["sha256_matches"] = current == info["sha256"]
    return meta


if __name__ == "__main__":
    # Run from the repository root:
    #   python -m stock_pred.src.data.load_data
    from stock_pred.src.config import load_config

    cfg = load_config()["data"]
    meta = download_prices(cfg["tickers"], cfg["start"], cfg["end"],
                           cfg["raw_dir"])
    print(json.dumps(meta, indent=2))
