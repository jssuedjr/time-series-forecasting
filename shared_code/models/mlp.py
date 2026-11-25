"""Simple one-hidden-layer MLP forecasting TSLA log returns, returning prices.

Uses only each supplied training window for normalization and fit.

Your code includes:
- Architecture: One hidden layer with 16 neurons by default.
- Activation: Hyperbolic tangent (tanh).
- Loss: Mean squared error, with its gradient calculated manually.
- Optimization: Adam, including first and second moments and bias correction.
- Preprocessing: Log returns standardized using the training window.
- Forecasting: Recursive multi-step predictions, converted back into prices.
"""

import numpy as np
from .model_interface import ForecastModel


class MLPModel(ForecastModel):
    def __init__(self, n_lags=20, hidden_size=16, learning_rate=0.001,
                 epochs=100, random_state=42):
        self.n_lags = int(n_lags)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.random_state = int(random_state)
        if min(self.n_lags, self.hidden_size, self.epochs) < 1 or self.learning_rate <= 0:
            raise ValueError('Invalid MLP hyperparameters')

    def fit(self, train):
        prices = np.asarray(train, dtype=float).reshape(-1)
        if len(prices) <= self.n_lags + 2 or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            raise ValueError('Need sufficient finite, strictly positive closing prices')
        returns = np.diff(np.log(prices))
        self.mean_ = float(returns.mean())
        self.std_ = float(returns.std())
        if self.std_ < 1e-10:
            self.std_ = 1.0
        r = (returns - self.mean_) / self.std_
        X = np.stack([r[i-self.n_lags:i] for i in range(self.n_lags, len(r))])
        y = r[self.n_lags:].reshape(-1, 1)
        rng = np.random.default_rng(self.random_state)
        self.W1 = rng.normal(0, np.sqrt(1 / self.n_lags), (self.n_lags, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = rng.normal(0, np.sqrt(1 / self.hidden_size), (self.hidden_size, 1))
        self.b2 = np.zeros((1, 1))
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        for t in range(1, self.epochs + 1):
            h = np.tanh(X @ self.W1 + self.b1)
            pred = h @ self.W2 + self.b2
            g = (2 / len(y)) * (pred - y)
            grads = [X.T @ ((g @ self.W2.T) * (1-h*h)),
                     ((g @ self.W2.T) * (1-h*h)).sum(axis=0, keepdims=True),
                     h.T @ g, g.sum(axis=0, keepdims=True)]
            for j, (p, grad) in enumerate(zip(params, grads)):
                m[j] = .9 * m[j] + .1 * grad
                v[j] = .999 * v[j] + .001 * grad * grad
                p -= self.learning_rate * (m[j] / (1-.9**t)) / (np.sqrt(v[j]/(1-.999**t)) + 1e-8)
        self.last_price_ = float(prices[-1])
        self.last_returns_ = r[-self.n_lags:].copy()
        return self

    def predict(self, horizon):
        if not hasattr(self, 'W1'):
            raise RuntimeError('Call fit() before predict()')
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        lag = self.last_returns_.copy()
        price = self.last_price_
        forecasts = []
        for _ in range(horizon):
            h = np.tanh(lag.reshape(1, -1) @ self.W1 + self.b1)
            scaled_return = float((h @ self.W2 + self.b2).item())
            log_return = scaled_return * self.std_ + self.mean_
            price *= np.exp(log_return)
            forecasts.append(price)
            lag = np.r_[lag[1:], scaled_return]
        return np.asarray(forecasts, dtype=float)
