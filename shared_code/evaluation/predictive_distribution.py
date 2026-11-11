import numpy as np

class EmpiricalPredictiveDistribution:
    """
    Estimate a future price distribution using historical
    out-of-sample forecast errors.

    Error convention: actual - predicted.
    """

    def __init__(self, forecast: float, errors):
        self.forecast = float(forecast)
        self.errors = np.asarray(errors, dtype=float)

        if not np.isfinite(self.forecast):
            raise ValueError("Forecast must be finite")

        if self.errors.ndim != 1 or len(self.errors) == 0:
            raise ValueError("Errors must be a nonempty 1D array")

        if not np.isfinite(self.errors).all():
            raise ValueError("Errors must be finite")

        # Shift each historical error by the final forecast.
        self.samples = self.forecast + self.errors

    def cdf(self, price):
        """Estimated probability that future price <= price."""
        values = np.asarray(price, dtype=float)

        if not np.isfinite(values).all():
            raise ValueError("Prices must be finite")

        sorted_samples = np.sort(self.samples)

        probabilities = (
            np.searchsorted(sorted_samples, values, side="right")
            / len(sorted_samples)
        )

        if values.ndim == 0:
            return float(probabilities)

        return probabilities

    def mean(self):
        return float(np.mean(self.samples))

    def variance(self):
        # Variance of the empirical distribution itself.
        return float(np.var(self.samples, ddof=0))

    def quantile(self, probability):
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1")

        # Inverse empirical CDF: smallest observed value
        # whose empirical CDF reaches the requested probability.
        sorted_samples = np.sort(self.samples)
        index = max(
            0,
            int(np.ceil(probability * len(sorted_samples))) - 1
        )

        return float(sorted_samples[index])

    def interval(self, level=0.95):
        if not 0 < level < 1:
            raise ValueError("Level must be between 0 and 1")

        alpha = 1 - level

        return (
            self.quantile(alpha / 2),
            self.quantile(1 - alpha / 2)
        )

    def sample(self, n=1000, random_state=None):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")

        rng = np.random.default_rng(random_state)

        return rng.choice(self.samples, size=n, replace=True)
