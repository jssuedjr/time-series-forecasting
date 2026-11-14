"""Stock-specific model wrappers (all forecasts in dollars)."""

import numpy as np
from scipy.stats import norm

from shared_code.models.random_walk import RandomWalk


class PriceRandomWalk(RandomWalk):
    """
    Random walk on log price without drift: log P_{T+h} = log P_T + e.

    Point forecast: P_T. Under a zero-drift log random walk, P_T is
    the MEDIAN of the predictive distribution (the project's point
    convention); the mean would be P_T * exp(h * sigma^2 / 2).

    Interval: P_T * exp(+/- z * sigma * sqrt(h)), with sigma the
    standard deviation of daily log returns in the training window
    (Gaussian, constant-volatility assumption; GARCH relaxes it).
    """

    def fit(self, train):
        super().fit(train)
        prices = np.asarray(train, dtype=float).reshape(-1)
        if (prices <= 0).any() or len(prices) < 3:
            raise ValueError("Need >= 3 positive prices")
        self.sigma = float(np.diff(np.log(prices)).std(ddof=1))
        return self

    def predict_interval(self, horizon: int, level: float = 0.95):
        z = norm.ppf(0.5 + level / 2)
        spread = z * self.sigma * np.sqrt(np.arange(1, horizon + 1))
        center = np.log(self.last_value)
        return np.exp(center - spread), np.exp(center + spread)
