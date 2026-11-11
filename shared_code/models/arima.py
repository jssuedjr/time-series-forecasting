import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from .model_interface import ForecastModel


class ARIMAModel(ForecastModel):
    """Univariate ARIMA forecasting model."""

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        trend: str | None = None
    ):
        self.order = order
        self.trend = trend
        self.results = None

    def fit(self, train):
        values = np.asarray(train, dtype=float)

        if values.ndim != 1:
            raise ValueError("Expected one-dimensional data")

        if not np.isfinite(values).all():
            raise ValueError("Training data must be finite")

        p, d, q = self.order

        if len(values) <= max(p, d, q) + 1:
            raise ValueError("Insufficient training observations")

        model = ARIMA(
            values,
            order=self.order,
            trend=self.trend
        )

        self.results = model.fit()

        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be positive")

        if self.results is None:
            raise RuntimeError("Call fit() before predict()")

        return np.asarray(
            self.results.forecast(steps=horizon),
            dtype=float
        )
