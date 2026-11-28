"""Check the fixed predictors and the two training windows without Kaggle data."""

import numpy as np
import pandas as pd
import pytest

from kaggle_comp.src.bella import fixed_forecast


@pytest.mark.parametrize("train_days, days_used", [(365, 365), (None, 400)])
def test_window_and_constant_predictors(monkeypatch, train_days, days_used):
    statsforecast = pytest.importorskip("statsforecast")
    dates = pd.date_range("2016-06-01", periods=416)
    index = pd.MultiIndex.from_product([[1, 2], ["BREAD"], dates],
                                       names=["store_nbr", "family", "date"])
    panel = index.to_frame(index=False)
    panel["sales"] = np.where(panel.store_nbr == 1, 2.0, 0.0)
    panel["onpromotion"] = panel.date.dt.dayofweek
    panel["holiday"] = (panel.date == dates[10]).astype(int)
    panel["closed"] = (panel.date == "2016-12-25").astype(int)
    # A flag that first changes during the forecast cannot be estimated at the cutoff.
    panel["event"] = (panel.date >= dates[400]).astype(int)
    cutoff = dates[399]
    future = panel.loc[panel.date > cutoff].drop(columns="sales")
    used_dates = []

    def check_model_input(self, df, h, X_df):
        used_dates.extend(df.ds)
        assert df.unique_id.unique().tolist() == ["1_BREAD"]
        assert "event" not in df and "event" not in X_df
        assert ("holiday" in df) == (train_days is None)
        assert "closed" in df and "closed" in X_df
        np.testing.assert_allclose(df.y, np.log1p(2.0))
        np.testing.assert_allclose(df.log_promo, np.log1p(df.ds.dt.dayofweek))
        return X_df[["unique_id", "ds"]].assign(AutoARIMA=np.log1p(3.0))

    monkeypatch.setattr(statsforecast.StatsForecast, "forecast", check_model_input)
    result = fixed_forecast(panel.loc[panel.date <= cutoff], future, train_days=train_days)
    assert len(used_dates) == days_used
    assert max(used_dates) == cutoff
    assert result.groupby("store_nbr").size().tolist() == [16, 16]
    np.testing.assert_allclose(result.loc[result.store_nbr == 1, "pred"], 3.0)
    np.testing.assert_array_equal(result.loc[result.store_nbr == 2, "pred"], 0.0)
