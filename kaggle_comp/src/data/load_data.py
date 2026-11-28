"""Load the daily sales table used by both Store Sales notebooks."""

from pathlib import Path

import pandas as pd

from .preprocess_data import CALENDAR_FLAGS, SERIES, add_predictors, store_calendar

KEYS = SERIES + ["date"]
SALES_COLUMNS = KEYS + ["id", "sales", "onpromotion"]


def _processed_panel(path):
    import pyarrow.parquet as pq

    available = set(pq.read_schema(path).names)
    missing = set(SALES_COLUMNS).difference(available)
    if missing:
        raise ValueError(f"{path.name} is missing {sorted(missing)}.")
    columns = SALES_COLUMNS + [
        c for c in ["split", "is_test", "city", "state", *CALENDAR_FLAGS] if c in available
    ]
    panel = pd.read_parquet(path, columns=columns)
    if "split" in panel:
        panel["is_test"] = panel["split"].eq("test")
    elif "is_test" not in panel:
        raise ValueError(f"{path.name} needs a split or is_test column.")
    return panel


def _raw_panel(directory):
    types = {"id": "Int32", "store_nbr": "int16", "family": "category", "onpromotion": "Int32"}
    train = pd.read_csv(directory / "train.csv", parse_dates=["date"], dtype=types)
    test = pd.read_csv(directory / "test.csv", parse_dates=["date"], dtype=types)
    if train["sales"].isna().any() or train["sales"].lt(0).any():
        raise ValueError("Historical sales must be known and nonnegative.")
    train_end = train["date"].max()
    if test["date"].min() != train_end + pd.Timedelta(days=1):
        raise ValueError("The first test date must follow the last training date.")
    rows = pd.concat([train, test], ignore_index=True)
    if not rows["id"].is_unique or rows.duplicated(KEYS).any():
        raise ValueError("Each sales row needs a unique id and store–family–date key.")
    pairs = rows[SERIES].drop_duplicates()
    dates = pd.date_range(rows["date"].min(), rows["date"].max())
    grid = pairs.merge(pd.DataFrame({"date": dates}), how="cross")
    panel = grid.merge(rows[SALES_COLUMNS], on=KEYS, how="left", validate="one_to_one")
    panel["is_test"] = panel["date"].gt(train_end)
    return panel


def load_sales_panel(data_dir):
    """Read processed.parquet or the raw Kaggle files. All added dates stay on the calendar."""
    directory = Path(data_dir)
    processed = directory / "processed.parquet"
    if processed.is_file():
        panel = _processed_panel(processed)
    elif all((directory / name).is_file() for name in ("train.csv", "test.csv")):
        panel = _raw_panel(directory)
    else:
        raise FileNotFoundError(f"Put processed.parquet or train.csv and test.csv in {directory}.")

    panel["date"] = pd.to_datetime(panel["date"])
    panel["family"] = panel["family"].astype("category")
    if "split" not in panel:
        panel["split"] = "train"
        panel.loc[panel["date"].ge("2017-07-31"), "split"] = "valid"
        panel.loc[panel["is_test"], "split"] = "test"
    panel["split"] = panel["split"].astype("category")
    if panel.duplicated(KEYS).any() or panel[KEYS].isna().any().any():
        raise ValueError("The panel has duplicate or missing store–family–date keys.")
    days = pd.date_range(panel["date"].min(), panel["date"].max())
    if not panel.groupby(SERIES, observed=True).size().eq(len(days)).all():
        raise ValueError("Each store and family needs one row per calendar day.")
    if panel.loc[panel["is_test"], "id"].isna().any():
        raise ValueError("A Kaggle test row is missing its id.")

    calendar = None
    events_path = directory / "holidays_events.csv"
    stores_path = directory / "stores.csv"
    store_columns = ["store_nbr", "city", "state"]
    if events_path.is_file() and (stores_path.is_file() or set(store_columns).issubset(panel)):
        events = pd.read_csv(events_path, parse_dates=["date"])
        if stores_path.is_file():
            stores = pd.read_csv(stores_path, usecols=store_columns)
        else:
            stores = panel[store_columns].drop_duplicates()
        calendar = store_calendar(stores, events, days)
    elif not set(CALENDAR_FLAGS).issubset(panel.columns):
        raise FileNotFoundError(
            f"This table needs stores.csv and holidays_events.csv in {directory} "
            "to build the holiday and event predictors."
        )
    return add_predictors(panel.drop(columns=["city", "state"], errors="ignore"), calendar)
