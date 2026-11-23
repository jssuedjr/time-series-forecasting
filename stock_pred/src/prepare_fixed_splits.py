"""Run from project root: python -m stock_pred.src.prepare_fixed_splits

Writes frozen benchmark input CSVs. Re-running after refreshing market data
preserves the same historical dates (assuming historical prices are unchanged).
"""
from pathlib import Path
import pandas as pd
from shared_code.backtest.splits import fixed_date_split
from stock_pred.src.split_config import (TRAIN_END_DATE, VALIDATION_END_DATE, TEST_END_DATE)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "stock_pred/data/tsla_historical_data.csv"
OUT = ROOT / "stock_pred/data/fixed_splits"


def load_prices(path=SOURCE):
    raw = pd.read_csv(path)
    raw["Date"] = pd.to_datetime(raw["Date"].astype(str).str[:10], format="%Y-%m-%d")
    prices = (raw.sort_values("Date").drop_duplicates("Date", keep="last")
              .set_index("Date")["Close"].astype(float))
    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("Closing prices must be positive and finite")
    return prices


def prepare():
    prices = load_prices()
    train, validation, test = fixed_date_split(
        prices, TRAIN_END_DATE, VALIDATION_END_DATE, TEST_END_DATE)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("validation", validation), ("test", test)):
        subset.rename("Close").to_csv(OUT / f"tsla_{name}.csv", index_label="Date")
        print(name, len(subset), subset.index.min().date(), subset.index.max().date())
    return train, validation, test


if __name__ == "__main__":
    prepare()
