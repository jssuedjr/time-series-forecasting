import numpy as np
import pandas as pd
import pytest

from kaggle_comp.src.data.load_data import load_sales_panel


def make_inputs(directory):
    days = pd.date_range("2016-12-23", "2017-01-03")
    rows = []
    for day in days:
        if day.day == 25:
            continue
        for store in (1, 2):
            for family in ("MEATS", "BOOKS"):
                promo = int(store == 1 and family == "MEATS" and day.day == 31) * 3
                rows.append((len(rows), day, store, family, promo, float(store == 1 or day.day == 31)))
    rows = pd.DataFrame(rows, columns=["id", "date", "store_nbr", "family", "onpromotion", "sales"])
    train = rows["date"].lt("2017-01-02")
    rows.loc[train].to_csv(directory / "train.csv", index=False)
    rows.loc[~train].drop(columns="sales").to_csv(directory / "test.csv", index=False)
    pd.DataFrame({"store_nbr": [1, 2], "city": ["Quito", "Cuenca"],
                  "state": ["Pichincha", "Azuay"]}).to_csv(directory / "stores.csv", index=False)
    descriptions = [
        ("2016-12-30", "Holiday", "Local", "Quito", "Local day", False),
        ("2016-12-31", "Holiday", "National", "Ecuador", "Moved day", True),
        ("2017-01-02", "Transfer", "National", "Ecuador", "Observed day", False),
        ("2016-12-31", "Work Day", "Regional", "Azuay", "Work day", False),
        ("2016-12-31", "Event", "National", "Ecuador", "Black Friday", False),
        ("2017-01-01", "Event", "National", "Ecuador", "Terremoto Manabi+1", False),
        ("2017-01-02", "Event", "National", "Ecuador", "Mundial de futbol", False),
    ]
    pd.DataFrame(descriptions, columns=["date", "type", "locale", "locale_name", "description",
                                        "transferred"]).to_csv(directory / "holidays_events.csv", index=False)


def test_payday_and_promotion_lags_cross_month_without_crossing_series(tmp_path):
    make_inputs(tmp_path)
    p = load_sales_panel(tmp_path).set_index(["store_nbr", "family", "date"])
    jan1 = p.loc[(1, "MEATS", "2017-01-01")]
    assert jan1["payday"] == 0 and jan1["after_payday"] == 1
    assert jan1["after_payday_weekend"] == 1 and jan1["closed"] == 1
    assert jan1["log_promo_lag1"] == np.log1p(3)
    assert p.loc[(1, "BOOKS", "2017-01-01"), "log_promo_lag1"] == 0
    assert p.loc[(2, "MEATS", "2017-01-01"), "log_promo_lag1"] == 0
    christmas = p.loc[(1, "MEATS", "2016-12-25")]
    assert christmas["sales"] == 0 and not christmas["observed"] and christmas["closed"] == 1
    assert not p.loc[(2, "MEATS", "2016-12-30"), "store_started"]
    assert p.loc[(2, "MEATS", "2016-12-31"), "store_started"]
    assert p.loc[p.is_test, "sales"].isna().all()


def test_observed_holiday_scopes_and_event_classes(tmp_path):
    make_inputs(tmp_path)
    p = load_sales_panel(tmp_path).set_index(["store_nbr", "family", "date"])
    quito = p.loc[(1, "BOOKS")]
    cuenca = p.loc[(2, "BOOKS")]
    assert quito.loc["2016-12-29", "before_holiday"] == 1
    assert cuenca.loc["2016-12-29", "before_holiday"] == 0
    assert quito.loc["2016-12-30", "hol_local"] == 1
    assert quito.loc["2016-12-31", "holiday"] == 0
    assert quito.loc["2016-12-31", "after_holiday"] == 1
    assert cuenca.loc["2016-12-31", "work_day"] == 1
    assert quito.loc["2016-12-31", "work_day"] == 0
    assert quito.loc["2016-12-31", "event_shopping"] == 1
    assert quito.loc["2017-01-01", "event"] == 1 and quito.loc["2017-01-01", "event_other"] == 0
    assert quito.loc["2017-01-02", "hol_national"] == 1 and quito.loc["2017-01-02", "event_sports"] == 1


def test_legacy_parquet_rebuilds_missing_flags_from_raw_calendar(tmp_path):
    make_inputs(tmp_path)
    from_csv = load_sales_panel(tmp_path)
    legacy = from_csv[["store_nbr", "family", "date", "id", "sales", "onpromotion", "is_test"]].copy()
    legacy.loc[legacy["id"].isna(), ["sales", "onpromotion"]] = np.nan
    legacy.to_parquet(tmp_path / "processed.parquet", index=False)
    (tmp_path / "train.csv").unlink()
    from_legacy = load_sales_panel(tmp_path)
    pd.testing.assert_frame_equal(from_csv, from_legacy, check_dtype=False, check_like=True)
    (tmp_path / "holidays_events.csv").unlink()
    with pytest.raises(FileNotFoundError, match="holidays_events.csv"):
        load_sales_panel(tmp_path)
