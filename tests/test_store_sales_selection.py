"""Checks for the two training windows and the predictor search."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture
def series_and_model(monkeypatch):
    models = pytest.importorskip("statsforecast.models")
    dates = pd.date_range("2017-01-01", periods=76)
    series = pd.DataFrame({"date": dates, "sales": np.arange(len(dates)) % 13})
    for day, name in enumerate(selection.WEEKDAYS, start=1):
        series[name] = (dates.weekday == day).astype(float)
    history, future = series.iloc[:60], series.iloc[60:]
    model = Mock()
    factory = Mock(return_value=model)
    monkeypatch.setattr(models, "AutoARIMA", factory)
    return history, future, model, factory


def test_forecast_uses_bellas_autoarima_settings_and_keeps_weekdays(series_and_model):
    history, future, model, factory = series_and_model
    model.model_ = {"code": 2, "arma": (0, 0, 0, 0, 7, 0, 0), "coef": {},
                    "bic": 2.0, "xreg": history[selection.WEEKDAYS].to_numpy(),
                    "residuals": np.linspace(-1, 1, len(history))}
    model.predict.return_value = {"mean": np.full(len(future), np.log1p(2.0))}

    pred, info = selection.forecast_one(history, future, selection.WEEKDAYS)

    factory.assert_called_once_with(season_length=7, ic="bic")
    model.fit.assert_called_once()
    np.testing.assert_allclose(model.fit.call_args.args[0], np.log1p(history.sales))
    np.testing.assert_array_equal(model.fit.call_args.kwargs["X"], history[selection.WEEKDAYS])
    np.testing.assert_array_equal(model.predict.call_args.kwargs["X"], future[selection.WEEKDAYS])
    np.testing.assert_allclose(pred, 2.0)
    assert info["method"] == "AutoARIMA"


@pytest.mark.parametrize(("failure", "expected_note"), [
    ("fit error", "Fit failed"), ("nonfinite forecast", "Invalid forecast"),
])
def test_forecast_uses_weekday_baseline_if_fit_or_forecast_fails(series_and_model, failure, expected_note):
    history, future, model, _ = series_and_model
    if failure == "fit error":
        model.fit.side_effect = ValueError("Fit failed")
    else:
        model.predict.return_value = {"mean": np.full(len(future), np.nan)}

    pred, info = selection.forecast_one(history, future, selection.WEEKDAYS)

    np.testing.assert_allclose(pred, selection.baseline_one(history, future))
    assert info["method"] == "weekday baseline"
    assert expected_note in info["note"]
