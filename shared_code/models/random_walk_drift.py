import numpy as np
from .model_interface import ForecastModel

class RandomWalkDrift(ForecastModel):
    """Random walk with estimated additive drift."""

    def fit(self, train):
        values = np.asarray(train, dtype=float)

        if values.ndim != 1 or len(values) < 2:
            raise ValueError("At least two observations required")

        if not np.isfinite(values).all():
            raise ValueError("Training data must be finite")

        self.last_value = values[-1]
        self.drift = np.diff(values).mean()

        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be positive")

        if not hasattr(self, "drift"):
            raise RuntimeError("Call fit() before predict()")

        steps = np.arange(1, horizon + 1)

        return self.last_value + self.drift * steps
