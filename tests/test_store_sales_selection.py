"""Checks for the two training windows and the predictor search."""

import numpy as np
import pandas as pd

from kaggle_comp.src import selection
from shared_code.backtest import kaggle as bt


def test_window_excludes_earthquake_and_masks_padded_sales():
    dates = pd.date_range("2016-04-01", "2017-06-28")
    christmas = (dates.month == 12) & (dates.day == 25)
    frame = pd.DataFrame({
        "date": dates, "sales": np.where(christmas, 0.0, 10.0),
        "store_started": True, "observed": ~christmas,
        "earthquake": (dates >= "2016-04-16") & (dates <= "2016-05-15"),
    })
    for days, expected_columns in [(365, []), (None, ["earthquake"])]:
        history, rows, future = selection._split_series(frame, bt.FOLDS["F1"], ["earthquake"], days)
        assert selection.usable_columns(history, ["earthquake"]) == expected_columns
        assert history.loc[history.date.eq("2016-12-25"), "sales"].isna().all()
        assert selection._training_log_sales(history)[history.date.eq("2016-12-25")][0] == np.log1p(10)
        assert list(future) == ["date", "earthquake"]
        assert len(rows) == bt.H


def test_stepwise_tests_delayed_payday_and_reuses_scores():
    groups = {k: selection.PREDICTOR_GROUPS[k] for k in ["payday", "day after payday", "trend"]}
    checked = []
    def evaluate(lists):
        assert set(lists).isdisjoint(checked)
        checked.extend(lists)
        scores = {}
        for label, cols in lists.items():
            # Here only the delayed effect improves the score. Trend helps by less than the threshold.
            scores[label] = 0.5 - 0.05 * ("after_payday" in cols) - 0.0001 * ("trend_years" in cols)
        return scores

    tried, path = selection._stepwise(groups, evaluate, 4, 0.0005, True)
    assert path == ["weekdays", "weekdays + payday + day after payday"]
    assert all("after_payday" not in cols or "payday" in cols for cols in tried.values())
    assert len(checked) == len(set(checked))
    assert len(tried) < 2**len(groups)


def test_stepwise_can_remove_predictors_and_swap_groups():
    groups = {"promotion": ["log_promo"], "event": ["event"], "event types": ["event_sports"],
              "trend": ["trend_years"]}
    scores = {"weekdays": 0.7, "weekdays + promotion": 0.61, "weekdays + event": 0.63,
              "weekdays + event types": 0.64, "weekdays + promotion + event": 0.6,
              "weekdays + promotion + event + trend": 0.56, "weekdays + event + trend": 0.54}
    def evaluate(lists):
        return {name: scores.get(name, 1.0) for name in lists}

    tried, path = selection._stepwise(groups, evaluate, 5, 0, True)
    assert path[-2:] == ["weekdays + promotion + event + trend", "weekdays + event + trend"]
    assert all(not {"event", "event_sports"}.issubset(cols) for cols in tried.values())
    assert any(s == {"event types", "trend"} for s in
               selection._neighbors(frozenset({"event", "trend"}), groups, False))


def test_pooled_score_uses_squared_errors():
    details = pd.DataFrame({
        "candidate": ["A", "A", "B", "B"], "period": ["F1", "F2", "F1", "F2"],
        "method": ["AutoARIMA"] * 4, "rmsle": [0.0, 0.8, 0.5, 0.5],
    })
    ranking = selection._rank(details, {"A": [], "B": ["payday"]})
    assert list(ranking.index) == ["B", "A"]
    assert np.isclose(ranking.loc["A", "selection_rmsle"], np.sqrt(0.32))


def test_failed_final_fit_retries_selected_orders_and_falls_back(monkeypatch):
    import pytest
    models = pytest.importorskip("statsforecast.models")
    from unittest.mock import Mock

    dates = pd.date_range("2017-01-01", periods=60)
    history = pd.DataFrame({"date": dates, "sales": np.arange(60) % 13})
    future = pd.DataFrame({"date": pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=16)})
    selected = Mock()
    selected.model_ = {"code": 2, "arma": (0, 1, 1, 0, 7, 1, 0), "coef": {"ma1": -0.8}}
    monkeypatch.setattr(models, "AutoARIMA", Mock(return_value=selected))
    factory = Mock(return_value={"code": 2})
    monkeypatch.setattr("statsforecast.arima.Arima", factory)

    pred, info = selection.forecast_one(history, future, [])
    retry = factory.call_args.kwargs
    assert retry["order"] == (0, 1, 1)
    assert retry["seasonal"] == {"order": (1, 0, 0), "period": 7}
    assert retry["method"] == "ML" and not retry["include_mean"] and not retry["include_drift"]
    assert retry["optim_control"]["gtol"] == 1e-5
    selected.predict.assert_not_called()
    np.testing.assert_allclose(pred, selection.baseline_one(history, future))
    assert info["method"] == "weekday baseline" and "ML retry returned code 2" in info["note"]


def test_successful_retry_uses_its_forecast_and_original_predictors(monkeypatch):
    import pytest
    models = pytest.importorskip("statsforecast.models")
    from unittest.mock import Mock

    history = pd.DataFrame({"date": pd.date_range("2017-01-01", periods=60),
                            "sales": np.arange(60) % 13, "payday": (np.arange(60) % 15 == 0)})
    future = pd.DataFrame({"date": pd.date_range("2017-03-02", periods=16),
                           "payday": (np.arange(16) % 15 == 0)})
    selected = Mock()
    selected.model_ = {"code": 2, "arma": (0, 0, 0, 0, 1, 0, 0),
                       "coef": {"intercept": 1.0, "ex_1": 0.2}}
    monkeypatch.setattr(models, "AutoARIMA", Mock(return_value=selected))
    fit = dict(selected.model_, code=0, bic=2.0, sigma2=0.2, xreg=history[["payday"]],
               residuals=np.linspace(-1, 1, 60))
    factory = Mock(return_value=fit)
    monkeypatch.setattr("statsforecast.arima.Arima", factory)
    selected.predict.return_value = {"mean": np.full(16, np.log1p(2.0))}

    pred, info = selection.forecast_one(history, future, ["payday"])
    assert factory.call_args.kwargs["include_mean"]
    assert selected.model_ is fit
    np.testing.assert_array_equal(factory.call_args.kwargs["xreg"], history[["payday"]])
    np.testing.assert_array_equal(selected.predict.call_args.kwargs["X"], future[["payday"]])
    np.testing.assert_allclose(pred, 2.0)
    assert info["method"] == "AutoARIMA" and "retried" in info["note"]
