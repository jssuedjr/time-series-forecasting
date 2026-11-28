"""ARMA-GARCH model on daily log returns, returning price forecasts.

Mean:      r_t = mu + phi_1 r_{t-1} + ... + phi_p r_{t-p} + eps_t   (ar_order=p)
Variance:  eps_t = sigma_t z_t,
           sigma_t^2 = omega + sum alpha_i eps_{t-i}^2 + sum beta_j sigma_{t-j}^2

Fitting the mean on log returns of prices is equivalent to ARIMA(p,1,0) on
log prices. Point forecast (dollars):  P_T * exp(sum of forecast mean returns),
i.e. the median price when the return shocks are symmetric.

predict_scale(h) returns the forecast standard deviation of the cumulative
log return through each horizon. The pipeline divides validation log-price
errors by this scale (standardized residuals) and rescales their empirical
quantiles at each forecast origin (filtered historical simulation), so
interval widths follow the GARCH volatility forecast.
"""
import warnings

import numpy as np
from arch import arch_model

from .model_interface import ForecastModel

# arch is numerically better behaved with returns in percent.
_PERCENT = 100.0


class ARIMAGARCHModel(ForecastModel):
    def __init__(self, ar_order: int = 0, garch_p: int = 1, garch_q: int = 1,
                 dist: str = "t", constant_mean: bool = True):
        # constant_mean only applies when ar_order == 0 (Constant vs Zero mean).
        if min(garch_p, garch_q) < 1 or ar_order < 0:
            raise ValueError("Require ar_order >= 0 and garch_p, garch_q >= 1")
        if dist not in {"normal", "t"}:
            raise ValueError("dist must be 'normal' or 't'")
        self.ar_order = int(ar_order)
        self.garch_p = int(garch_p)
        self.garch_q = int(garch_q)
        self.dist = dist
        self.constant_mean = bool(constant_mean)
        self.results = None

    def fit(self, train):
        prices = np.asarray(train, dtype=float).reshape(-1)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Prices must be finite and strictly positive")
        if len(prices) < 50 + self.ar_order:
            raise ValueError("Insufficient observations for GARCH estimation")

        returns = _PERCENT * np.diff(np.log(prices))
        if self.ar_order == 0:
            mean = "Constant" if self.constant_mean else "Zero"
        else:
            mean = "AR"  # arch always includes an intercept for AR means
        model = arch_model(returns, mean=mean, lags=self.ar_order, vol="GARCH",
                           p=self.garch_p, q=self.garch_q, dist=self.dist,
                           rescale=False)

        with warnings.catch_warnings():
            # Occasional convergence warnings on short rolling windows;
            # the fitted values are still used, but failures are caught below.
            warnings.simplefilter("ignore")
            self.results = model.fit(disp="off", show_warning=False)

        if not np.isfinite(self.results.params).all():
            raise RuntimeError("GARCH estimation returned non-finite parameters")
        self.last_price_ = float(prices[-1])
        self._forecast_cache = {}
        return self

    def _forecast(self, horizon):
        if self.results is None:
            raise RuntimeError("Call fit() before predict()")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if horizon not in self._forecast_cache:
            f = self.results.forecast(horizon=int(horizon), reindex=False)
            mean = f.mean.to_numpy()[-1] / _PERCENT
            # Variance of each one-step return; GARCH return shocks are
            # uncorrelated, so cumulative variance is (approximately, for
            # AR means) their running sum.
            variance = f.variance.to_numpy()[-1] / _PERCENT ** 2
            self._forecast_cache[horizon] = (mean, variance)
        return self._forecast_cache[horizon]

    def predict(self, horizon: int) -> np.ndarray:
        mean, _ = self._forecast(horizon)
        return self.last_price_ * np.exp(np.cumsum(mean))

    def predict_scale(self, horizon: int) -> np.ndarray:
        """Forecast std. dev. of the cumulative log return at each horizon."""
        _, variance = self._forecast(horizon)
        return np.sqrt(np.cumsum(variance))

    def information_criteria(self) -> dict:
        if self.results is None:
            raise RuntimeError("Call fit() first")
        r = self.results
        return {"AIC": float(r.aic), "BIC": float(r.bic),
                "log_likelihood": float(r.loglikelihood),
                "parameters": int(len(r.params))}
