import numpy as np
import pandas as pd
from matplotlib.figure import Figure

####### Residual diagnostics #######
'''
Residual diagnostics primarily examine whether the fitted model has 
adequately captured the structure in the observed data.
'''

def plot_residual_acf_pacf(
    # Examine serial correlation remaining in the residuals
    residuals: pd.Series | np.ndarray,
    lags: int = 20
):
    ...

def ljung_box_test(
    # Test whether autocorrelations through specified lags are jointly zero
    residuals: pd.Series | np.ndarray,
    lags: int | list[int] = 10
) -> pd.DataFrame:
    ...

def plot_residual_distribution(
    # Examine skewness, outliers, and distributional shape
    residuals: pd.Series | np.ndarray,
    bins: int = 30
):
    ...

def test_heteroskedasticity(
    # Test for time-varying conditional variance, e.g. ARCH effects
    residuals: pd.Series | np.ndarray,
    lags: int = 10
) -> pd.DataFrame:
    ...


####### Forecast diagnostics #######
'''
These functions examine how well your models predict observations that 
were not available when the forecasts were generated.

I'd design them around your standardized backtesting results DataFrame.
'''

def plot_error_by_horizon(
    # helps determine whether forecast accuracy deteriorates 
    # as the prediction horizon increases.
    results: pd.DataFrame,
    metric: str = "rmse"
):
    ...

def calculate_forecast_bias(
    # measures systematic overprediction or underprediction
    results: pd.DataFrame,
    group_by: str | list[str] | None = None
) -> pd.DataFrame:
    ...

def plot_rolling_forecast_error(
    # helps identify periods when a model's predictive performance becomes worse. 
    # The rolling window here refers to the evaluation window, 
    # not necessarily the training window used during backtesting.
    results: pd.DataFrame,
    window: int = 20,
    metric: str = "rmse"
):
    ...

def plot_forecast_error_distribution(
    results: pd.DataFrame,
    bins: int = 30
):
    ...



####### Time-series diagnostics #######
'''
These functions operate on the observed time series rather than on model forecasts.
'''

def adf_test(
    # Hypothesis test whose null is "The series has a unit root"
    series: pd.Series | np.ndarray,
    regression: str = "c",
    autolag: str = "AIC"
) -> pd.DataFrame:
    ...

def kpss_test(
    # Hypothesis test whose null is "The series is stationary around a level or 
    # deterministic trend, depending on specification"
    series: pd.Series | np.ndarray,
    regression: str = "c",
    nlags: str | int = "auto"
) -> pd.DataFrame:
    ...

def plot_stationarity(
    # initially plot the original series, rolling mean, and rolling standard deviation. 
    # These are exploratory visualizations, not formal stationarity tests.
    series: pd.Series,
    window: int = 20
):
    ...

def plot_series_acf_pacf(
    series: pd.Series | np.ndarray,
    lags: int = 20
):
    ...