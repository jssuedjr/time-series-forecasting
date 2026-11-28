import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

from shared_code.backtest.backtest import univariate_backtest
from shared_code.evaluation.metrics import interval_score
from shared_code.models.arima_garch import ARIMAGARCHModel
from stock_pred.src.full_stock_pred_pipeline import (
    build_predictive_distribution,
    run_forecasting_pipeline,
)


def garch_prices(n=500, seed=0):
    """Simulated GARCH(1,1) log returns turned into positive prices."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 1e-5, 0.08, 0.9
    var, r = omega / (1 - alpha - beta), []
    for _ in range(n):
        eps = np.sqrt(var) * rng.standard_normal()
        r.append(eps)
        var = omega + alpha * eps ** 2 + beta * var
    return 100 * np.exp(np.cumsum(r))


def test_fit_predict_shapes_and_positive_scale():
    model = ARIMAGARCHModel(dist="normal").fit(garch_prices())
    forecast = model.predict(3)
    scale = model.predict_scale(3)
    assert forecast.shape == scale.shape == (3,)
    assert np.isfinite(forecast).all() and (forecast > 0).all()
    # Cumulative volatility grows with the horizon.
    assert (scale > 0).all() and (np.diff(scale) > 0).all()


def test_rejects_non_positive_prices():
    prices = garch_prices()
    prices[10] = -1.0
    with pytest.raises(ValueError):
        ARIMAGARCHModel().fit(prices)


def test_backtest_records_pred_scale_only_for_scaled_models():
    s = pd.Series(garch_prices(320), index=pd.bdate_range("2020-01-01", periods=320))
    bt = univariate_backtest(s, lambda: ARIMAGARCHModel(dist="normal"),
                             initial_train_size=300, strategy="rolling", window_size=300)
    assert "pred_scale" in bt.columns and (bt["pred_scale"] > 0).all()


def test_scaled_distribution_is_multiplicative():
    z = np.array([-2.0, 0.0, 1.0])
    dist = build_predictive_distribution(100.0, None, standardized_errors=z, scale=0.05)
    np.testing.assert_allclose(np.sort(dist.samples), 100 * np.exp(z * 0.05))
    with pytest.raises(ValueError):
        build_predictive_distribution(100.0, None, standardized_errors=z, scale=None)


def test_pipeline_uses_volatility_scaled_intervals():
    s = pd.Series(garch_prices(700), index=pd.bdate_range("2020-01-01", periods=700))
    res = run_forecasting_pipeline(s, ARIMAGARCHModel, model_kwargs={"dist": "normal"},
                                   window_size=300, train_end=400, validation_end=550)
    width = res.test["interval_upper"] - res.test["interval_lower"]
    assert res.standardized_errors is not None
    assert width.std() > 0  # time-varying, unlike residual-shift intervals
    assert res.forecast["pred_scale"].iloc[0] > 0


def test_interval_score():
    y = np.array([10.0, 10.0, 10.0])
    lower = np.array([9.0, 11.0, 5.0])
    upper = np.array([11.0, 12.0, 9.0])
    # widths 2, 1, 4; misses: 1 below lower, 1 above upper; 2/alpha = 40 at 95%
    assert interval_score(y, lower, upper, 0.95) == pytest.approx((2 + (1 + 40) + (4 + 40)) / 3)
