"""Calendar and promotion columns for daily store–family forecasts."""

import numpy as np
import pandas as pd

SERIES = ["store_nbr", "family"]
STORE_DATE = ["store_nbr", "date"]
HOLIDAY_FLAGS = ["hol_national", "hol_regional", "hol_local"]
EVENT_FLAGS = ["event_shopping", "event_sports", "event_other"]
CALENDAR_FLAGS = HOLIDAY_FLAGS + ["holiday", "event", "work_day"]


def store_calendar(stores, events, dates):
    """Match recorded holidays and events to the stores where they apply."""
    observed = events.loc[~events["transferred"]].copy()
    is_holiday = observed["type"].isin(["Holiday", "Transfer", "Additional", "Bridge"])
    is_event = observed["type"].eq("Event")
    description = observed["description"].str.lower()
    shopping = description.isin(["black friday", "cyber monday", "dia de la madre"])
    sports = description.str.contains("mundial", na=False)
    quake = description.str.startswith("terremoto", na=False)
    observed["holiday"] = is_holiday
    observed["event"] = is_event
    observed["work_day"] = observed["type"].eq("Work Day")
    observed["event_shopping"] = is_event & shopping
    observed["event_sports"] = is_event & sports
    observed["event_other"] = is_event & ~(shopping | sports | quake)
    for scope, column in zip(["National", "Regional", "Local"], HOLIDAY_FLAGS):
        observed[column] = is_holiday & observed["locale"].eq(scope)

    # Include one extra day at either end for holiday neighbors.
    dates = pd.date_range(dates.min() - pd.Timedelta(days=1),
                          dates.max() + pd.Timedelta(days=1))
    calendars = []
    flags = CALENDAR_FLAGS + EVENT_FLAGS
    for store in stores.itertuples(index=False):
        applies = (observed["locale"].eq("National")
                   | (observed["locale"].eq("Regional") & observed["locale_name"].eq(store.state))
                   | (observed["locale"].eq("Local") & observed["locale_name"].eq(store.city)))
        calendar = observed.loc[applies].groupby("date")[flags].max()
        calendar = calendar.reindex(dates, fill_value=False).astype("int8")
        calendar["before_holiday"] = calendar["holiday"].shift(-1, fill_value=0)
        calendar["after_holiday"] = calendar["holiday"].shift(1, fill_value=0)
        calendars.append(calendar.rename_axis("date").reset_index().assign(store_nbr=store.store_nbr))
    return pd.concat(calendars, ignore_index=True)


def _daily_features(dates):
    calendar = pd.DataFrame({"date": dates})
    day = calendar["date"].dt
    previous = (calendar["date"] - pd.Timedelta(days=1)).dt
    calendar["payday"] = (day.day.eq(15) | day.is_month_end).astype("int8")
    calendar["after_payday"] = (previous.day.eq(15) | previous.is_month_end).astype("int8")
    calendar["after_payday_weekend"] = calendar["after_payday"] * day.weekday.ge(5).astype("int8")
    calendar["earthquake"] = calendar["date"].between("2016-04-16", "2016-05-15").astype("int8")
    calendar["closed"] = ((day.month.eq(1) & day.day.eq(1))
                          | (day.month.eq(12) & day.day.eq(25))).astype("int8")
    for weekday in range(1, 7):
        calendar[f"dow_{weekday}"] = day.weekday.eq(weekday).astype("int8")
    elapsed = (calendar["date"] - dates.min()).dt.days / 365.25
    calendar["trend_years"] = elapsed.astype("float32")
    for k in (1, 2):
        angle = 2 * np.pi * k * elapsed
        calendar[f"sin{k}"] = np.sin(angle).astype("float32")
        calendar[f"cos{k}"] = np.cos(angle).astype("float32")
    return calendar


def add_predictors(panel, calendar):
    """Keep notebook 02's sales for the fixed model and flag added dates for selection."""
    panel = panel.sort_values(SERIES + ["date"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    panel["observed"] = panel["id"].notna() & ~panel["is_test"]
    added_days = panel["id"].isna() & ~panel["is_test"]
    panel.loc[added_days, ["sales", "onpromotion"]] = 0
    if panel.loc[panel["observed"], "sales"].isna().any() or panel["onpromotion"].isna().any():
        raise ValueError("Observed sales and scheduled promotions must have values.")
    if panel.loc[panel["is_test"], "sales"].notna().any():
        raise ValueError("Kaggle test sales must be unknown.")
    if calendar is None:
        # Notebook 02 already matched holidays and events to each store.
        calendar = panel[STORE_DATE + CALENDAR_FLAGS].drop_duplicates().sort_values(STORE_DATE)
        by_store = calendar.groupby("store_nbr", observed=True)["holiday"]
        calendar["before_holiday"] = by_store.shift(-1, fill_value=0)
        calendar["after_holiday"] = by_store.shift(1, fill_value=0)
    all_flags = [c for c in calendar.columns if c not in STORE_DATE]
    calendar[all_flags] = calendar[all_flags].astype("int8")
    panel = panel.drop(columns=all_flags, errors="ignore").merge(
        calendar, on=STORE_DATE, how="left", validate="many_to_one", sort=False
    )
    if panel["holiday"].isna().any():
        raise ValueError("Every store and date needs a holiday calendar.")
    daily = _daily_features(dates)
    panel = panel.merge(daily, on="date", how="left", validate="many_to_one", sort=False)
    panel["log_promo"] = np.log1p(panel["onpromotion"].astype("float64"))
    panel["log_promo_lag1"] = panel.groupby(SERIES, observed=True, sort=False)["log_promo"].shift(
        1, fill_value=0.0
    )

    # A store has started once any family records a positive sale on or before this date.
    active_days = panel.groupby(STORE_DATE, observed=True)["sales"].max().gt(0)
    started = active_days.groupby(level="store_nbr").cummax().rename("store_started")
    return panel.join(started, on=STORE_DATE)
