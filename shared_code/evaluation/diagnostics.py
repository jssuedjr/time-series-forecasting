"""Shared time-series and one-step forecast-error diagnostics."""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from shared_code.validation import validate_series, validate_backtest_results


def adf_test(series, regression='c', autolag='AIC')->pd.DataFrame:
    x=validate_series(series, min_length=10)
    result=adfuller(x,regression=regression,autolag=autolag)
    return pd.DataFrame([{'adf_stat':result[0],'adf_p':result[1],'adf_lags':result[2]}])

def kpss_test(series,regression='c',nlags='auto')->pd.DataFrame:
    x=validate_series(series,min_length=10)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result=kpss(x,regression=regression,nlags=nlags)
    return pd.DataFrame([{'kpss_stat':result[0],'kpss_p':result[1],'kpss_lags':result[2]}])

def ljung_box_test(residuals,lags=10)->pd.DataFrame:
    x=validate_series(residuals,min_length=3)
    return acorr_ljungbox(x,lags=lags,return_df=True)

def test_heteroskedasticity(residuals,lags=10)->pd.DataFrame:
    x=validate_series(residuals,min_length=lags+3)
    stat,p,_,_=het_arch(x-x.mean(),nlags=lags)
    return pd.DataFrame([{'ARCH_LM_stat':stat,'ARCH_pvalue':p,'lags':lags}])

def describe_series(series,name)->dict:
    """Training-only diagnostic summary; ADF/KPSS and Ljung-Box/ARCH."""
    x=pd.Series(validate_series(series,min_length=20))
    adf=adf_test(x).iloc[0]
    try: kp=kpss_test(x).iloc[0]['kpss_p']
    except ValueError: kp=np.nan
    return {'series':name,'n':len(x),'mean':x.mean(),'std':x.std(),
            'skew':x.skew(),'excess_kurtosis':x.kurt(),
            'adf_stat':adf['adf_stat'],'adf_p':adf['adf_p'],'adf_lags':adf['adf_lags'],
            'kpss_p':kp,'lb10_p':float(ljung_box_test(x,lags=[10])['lb_pvalue'].iloc[0]),
            'arch10_p':float(test_heteroskedasticity(x,lags=10)['ARCH_pvalue'].iloc[0])}

def training_diagnostic_table(train_prices)->pd.DataFrame:
    p=pd.Series(train_prices).astype(float)
    if (p<=0).any():raise ValueError('Log returns require strictly positive prices')
    r=np.log(p).diff().dropna()
    return pd.DataFrame([describe_series(p,'training price'),
                         describe_series(p.diff().dropna(),'training price difference'),
                         describe_series(r,'training log returns')]).set_index('series')

def forecast_error_diagnostics(errors)->pd.DataFrame:
    x=validate_series(errors,min_length=3)
    n=len(x)
    out={'n_errors':n,'mean_error':float(x.mean()),'std_error':float(x.std(ddof=1)),
         't_pvalue':float(stats.ttest_1samp(x,0).pvalue)}
    fitted=sm.OLS(x,np.ones((n,1))).fit(cov_type='HAC',cov_kwds={'maxlags':min(5,n-1)})
    out['HAC_pvalue']=float(fitted.pvalues[0])
    for lag in (5,10):
        if lag<n:out[f'LB_{lag}_pvalue']=float(ljung_box_test(x,lags=[lag])['lb_pvalue'].iloc[0])
    if n>15:out['ARCH_pvalue']=float(test_heteroskedasticity(x,lags=min(10,(n-1)//2))['ARCH_pvalue'].iloc[0])
    return pd.DataFrame([out])

def calculate_forecast_bias(results,group_by=None)->pd.DataFrame:
    data=validate_backtest_results(results)
    groups=[group_by] if isinstance(group_by,str) else list(group_by or [])
    if not groups:return pd.DataFrame([{'mean_error':float(data['error'].mean())}])
    return data.groupby(groups,as_index=False).agg(mean_error=('error','mean'))

def plot_series_acf_pacf(series,lags=20):
    x=validate_series(series,min_length=5); lags=min(lags,len(x)//2-1)
    fig,axes=plt.subplots(1,2,figsize=(12,3),constrained_layout=True)
    plot_acf(x,lags=lags,ax=axes[0]);plot_pacf(x,lags=lags,ax=axes[1],method='ywm')
    return fig,axes

def plot_residual_acf_pacf(residuals,lags=20):
    return plot_series_acf_pacf(residuals,lags)

def plot_residual_distribution(residuals,bins=30):
    x=validate_series(residuals)
    fig,ax=plt.subplots(figsize=(8,3));ax.hist(x,bins=bins,density=True)
    ax.set_title('Residual distribution');return fig,ax

def plot_forecast_error_distribution(results,bins=30):
    data=validate_backtest_results(results)
    fig,ax=plt.subplots(figsize=(8,3));ax.hist(data['error'],bins=bins)
    ax.set_title('Forecast-error distribution');return fig,ax

def plot_rolling_forecast_error(results,window=20,metric='rmse'):
    data=validate_backtest_results(results).sort_values('target_date')
    e=data['error'];values=(e**2).rolling(window).mean()**0.5 if metric=='rmse' else e.abs().rolling(window).mean()
    if metric not in {'rmse','mae'}:raise ValueError('metric must be rmse or mae')
    fig,ax=plt.subplots(figsize=(10,3));ax.plot(data['target_date'],values)
    ax.set_title(f'Rolling {window}-forecast {metric.upper()}');return fig,ax

def plot_error_by_horizon(results,metric='rmse'):
    data=validate_backtest_results(results)
    agg=data.groupby('horizon')['error'].apply(lambda e:float(np.sqrt(np.mean(e**2))) if metric=='rmse' else float(np.mean(np.abs(e))))
    if metric not in {'rmse','mae'}:raise ValueError('metric must be rmse or mae')
    fig,ax=plt.subplots(figsize=(7,3));agg.plot.bar(ax=ax);ax.set_ylabel(metric.upper());return fig,ax

def plot_stationarity(series,window=20):
    x=pd.Series(series).astype(float)
    fig,ax=plt.subplots(figsize=(10,4));x.plot(ax=ax,label='Series');x.rolling(window).mean().plot(ax=ax,label='Rolling mean');x.rolling(window).std().plot(ax=ax,label='Rolling std');ax.legend();return fig,ax
