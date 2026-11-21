"""Reusable plots for canonical backtest results and predictive distributions."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from shared_code.validation import validate_backtest_results


def plot_price_return_exploration(prices,volatility_window=20):
    log_returns=np.log(prices).diff().dropna()
    fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
    prices.plot(ax=axes[0],title='TSLA closing price')
    log_returns.plot(ax=axes[1],title='Daily log returns')
    log_returns.rolling(volatility_window).std().plot(ax=axes[2],title=f'{volatility_window}-session rolling standard deviation of log returns')
    return fig,axes

def plot_training_acf_pacf(train_prices,lags=30):
    r=np.log(train_prices).diff().dropna()
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,s,fn,title in [(axes[0,0],train_prices,plot_acf,'Price ACF'),(axes[0,1],train_prices,plot_pacf,'Price PACF'),
                          (axes[1,0],r,plot_acf,'Log-return ACF'),(axes[1,1],r,plot_pacf,'Log-return PACF')]:
        kw={'method':'ywm'} if fn is plot_pacf else {}
        fn(s,lags=min(lags,len(s)//2-1),ax=ax,**kw);ax.set_title(title)
    fig.suptitle('Training data: autocorrelation and partial autocorrelation')
    return fig,axes

def plot_forecasts(results):
    return plot_backtest_comparison(results)

def plot_backtest_comparison(validation,test):
    fig,axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
    for ax,bt,label in zip(axes,[validation,test],['Validation','Held-out test']):
        validate_backtest_results(bt)
        ax.plot(bt['target_date'],bt['y_true'],label='Actual close')
        ax.plot(bt['target_date'],bt['y_pred'],label='1-step model forecast',alpha=.8)
        ax.set(title=f'{label}: rolling-window forecasts',ylabel='Closing price ($)');ax.legend()
    return fig,axes

def plot_backtest_errors(validation,test):
    fig,axes=plt.subplots(2,1,figsize=(12,6),constrained_layout=True)
    for ax,bt,label in zip(axes,[validation,test],['Validation','Held-out test']):
        validate_backtest_results(bt)
        ax.plot(bt['target_date'],bt['error']);ax.axhline(0,linestyle='--',color='black',linewidth=1)
        ax.set(title=f'{label}: forecast error (actual − predicted)',ylabel='Error ($)')
    return fig,axes

def plot_validation_error_diagnostics(validation):
    errors=validate_backtest_results(validation).set_index('target_date')['error'].astype(float)
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    errors.plot(ax=axes[0,0],title='Validation forecast errors');axes[0,0].axhline(0,linestyle='--',color='black',linewidth=1)
    axes[0,1].hist(errors,bins=30,density=True);axes[0,1].set_title('Validation error histogram')
    plot_acf(errors,lags=min(20,len(errors)//2-1),ax=axes[1,0]);axes[1,0].set_title('Error ACF')
    plot_acf(errors**2,lags=min(20,len(errors)//2-1),ax=axes[1,1]);axes[1,1].set_title('Squared-error ACF (volatility clustering)')
    return fig,axes

def plot_test_intervals_and_errors(check,level=.95):
    validate_backtest_results(check)
    fig,axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
    axes[0].plot(check['target_date'],check['y_true'],label='Actual close',linewidth=1.5)
    axes[0].plot(check['target_date'],check['y_pred'],label='Original model forecast',alpha=.8)
    axes[0].plot(check['target_date'],check['predictive_mean'],label='Bias-adjusted empirical mean',alpha=.7)
    axes[0].fill_between(check['target_date'],check['interval_lower'],check['interval_upper'],alpha=.18,label=f'Empirical {level:.0%} interval')
    axes[0].set(title='Held-out forecasts and validation-calibrated prediction intervals',ylabel='Closing price ($)');axes[0].legend()
    axes[1].plot(check['target_date'],check['error'],label='Model forecast error')
    axes[1].axhline(0,linestyle='--',color='black',linewidth=1)
    axes[1].set(title='Held-out one-step forecast errors',ylabel='Actual − predicted ($)')
    return fig,axes

def plot_predictive_cdf(distribution,ax=None):
    if ax is None:_,ax=plt.subplots(figsize=(9,5))
    prices=np.sort(distribution.samples);probabilities=np.arange(1,len(prices)+1)/len(prices)
    ax.step(prices,probabilities,where='post',label='Empirical predictive CDF')
    ax.axvline(distribution.forecast,linestyle='--',label='Point forecast')
    ax.set(xlabel='Possible future closing price ($)',ylabel='Cumulative probability',title='Empirical predictive distribution');ax.legend()
    return ax

def plot_predictive_distribution(distribution,level=.95):
    point=distribution.forecast;mean=distribution.mean();lower,upper=distribution.interval(level)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    prices=np.sort(distribution.samples)
    axes[0].step(prices,np.arange(1,len(prices)+1)/len(prices),where='post',label='Empirical CDF')
    axes[0].axvline(point,linestyle='--',color='black',label='Model point forecast')
    axes[0].axvline(mean,linestyle='-.',label='Predictive mean')
    axes[0].axvline(lower,linestyle=':',color='gray');axes[0].axvline(upper,linestyle=':',color='gray')
    axes[0].set(title='Shifted validation-error ECDF',xlabel='Next closing price ($)',ylabel='Cumulative probability');axes[0].legend()
    axes[1].hist(distribution.samples,bins=30,density=True)
    axes[1].axvline(point,linestyle='--',color='black',label='Model point forecast')
    axes[1].axvline(mean,linestyle='-.',label='Predictive mean')
    axes[1].set(title='Empirical predictive samples',xlabel='Next closing price ($)');axes[1].legend()
    return fig,axes

def plot_residuals(results):
    data=validate_backtest_results(results)
    fig,ax=plt.subplots(figsize=(10,3));ax.plot(data['target_date'],data['error']);ax.axhline(0,linestyle='--',color='black');return fig,ax

def plot_acf_pacf(residuals):
    from shared_code.evaluation.diagnostics import plot_residual_acf_pacf
    return plot_residual_acf_pacf(residuals)
