
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

import pandas_market_calendars as mcal

def latest_completed_nyse_session():
    now = pd.Timestamp.now(tz="America/New_York")
    nyse = mcal.get_calendar("NYSE")

    schedule = nyse.schedule(
        start_date=(now - pd.Timedelta(days=14)).date(),
        end_date=now.date(),
    )

    completed = schedule.loc[
        schedule["market_close"] <= now.tz_convert("UTC")
    ]

    if completed.empty:
        raise RuntimeError("No completed NYSE session found")

    return pd.Timestamp(completed.index[-1]).normalize()

def update_market_data(ticker="TSLA", start_date="2021-11-11",interval="1d"):
    """Fetch and persist daily historical prices."""

    df = yf.download(
        ticker,
        start=start_date,
        interval=interval,
        auto_adjust=False,
        multi_level_index=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("No market data returned")

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    # For the first prospective forecast, use completed sessions
    # through November 19, 2026 only.
    last_allowed_date = latest_completed_nyse_session()
    df = df.loc[
        df["Date"] <= last_allowed_date
    ].copy()

    df = (
        df.drop_duplicates(subset="Date", keep="last")
          .sort_values("Date")
          .reset_index(drop=True)
    )

    path = DATA_DIR / f"{ticker.lower()}_historical_data.csv"

    # Preserve previously downloaded observations.
    if path.exists():
        old = pd.read_csv(path)

        old["Date"] = pd.to_datetime(
            old["Date"].astype(str).str[:10]
        )

        old = old.loc[
            old["Date"] <= last_allowed_date
        ].copy()

        df = pd.concat([old, df], ignore_index=True)

        df = (
            df.drop_duplicates(subset="Date", keep="last")
            .sort_values("Date")
            .reset_index(drop=True)
        )

    # Verify that the most recent observation is the
    # latest completed NYSE trading session.

    latest_data_date = pd.Timestamp(df["Date"].max()).normalize()

    if latest_data_date != last_allowed_date:
        raise RuntimeError(
            "TSLA data is not current.\n"
            f"Latest completed NYSE session: {last_allowed_date.date()}\n"
            f"Latest available TSLA observation: {latest_data_date.date()}\n"
            "Yahoo Finance may not have published the latest "
            "completed daily candle yet. Try again later."
        )

    print(
        "Verified latest completed TSLA session:",
        latest_data_date.date()
    )

    df.to_csv(path, index=False)

    run_time = datetime.now(ZoneInfo("America/New_York"))

    return df, run_time
