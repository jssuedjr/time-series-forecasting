import numpy as np
from .model_interface import ForecastModel

class RandomWalk(ForecastModel):

    def fit(self, train):
        values = np.asarray(train, dtype=float).reshape(-1)

        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Training data must be nonempty and finite.")

        self.last_value = values[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if horizon < 1:
            raise ValueError("horizon must be positive")

        if not hasattr(self, "last_value"):
            raise RuntimeError("Call fit() before predict().")

        return np.full(horizon, self.last_value)
