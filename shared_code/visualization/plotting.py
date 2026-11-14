import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_forecasts(
    results: pd.DataFrame
):
    ...

def plot_residuals(
    results: pd.DataFrame
):
    ...

def plot_acf_pacf(
    residuals: np.ndarray
):
    ...



def plot_predictive_cdf(distribution, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    prices = np.sort(distribution.samples)

    probabilities = (
        np.arange(1, len(prices) + 1) / len(prices)
    )

    ax.step(
        prices,
        probabilities,
        where="post",
        label="Empirical predictive CDF"
    )

    ax.axvline(
        distribution.forecast,
        linestyle="--",
        label="Point forecast"
    )

    ax.set_xlabel("Possible future closing price ($)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Empirical predictive distribution")
    ax.legend()

    return ax
