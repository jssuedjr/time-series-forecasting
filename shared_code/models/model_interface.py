
from abc import ABC, abstractmethod
import numpy as np


class ForecastModel(ABC):
    """Abstract interface for univariate forecasting models."""

    @abstractmethod
    def fit(self, train):
        """Fit the model using training data."""
        pass

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Return forecasts for the next horizon observations."""
        pass

#################################################################
#################################################################

class UpdatableForecastModel(ABC):
    """
    Optional interface for models supporting
    updates without full parameter refitting.
    """

    @abstractmethod
    def update(self, new_data):
        """
        Incorporate newly observed data without
        re-estimating model parameters.
        """
        raise NotImplementedError

