"""Forecast accuracy and interval metrics for canonical backtest tables."""
import numpy as np
import pandas as pd
from shared_code.validation import validate_numeric_pair, validate_forecast_results


def mse(y_true, y_pred) -> float:
    y, p = validate_numeric_pair(y_true, y_pred)
    return float(np.mean((y-p)**2))

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    y, p = validate_numeric_pair(y_true, y_pred)
    return float(np.mean(np.abs(y-p)))

def rmsle(y_true, y_pred) -> float:
    y, p = validate_numeric_pair(y_true, y_pred, nonnegative=True)
    return float(np.sqrt(np.mean((np.log1p(y)-np.log1p(p))**2)))

def mase(y_true, y_pred, training_series=None, seasonality=1) -> float:
    """MASE uses a naive seasonal-difference scale from *training* data."""
    if training_series is None:
        raise ValueError('mase requires training_series for a leakage-free scale')
    train = np.asarray(training_series, dtype=float)
    if seasonality < 1 or len(train) <= seasonality:
        raise ValueError('Invalid training length or seasonality')
    scale = np.mean(np.abs(train[seasonality:] - train[:-seasonality]))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError('MASE training scale must be finite and nonzero')
    return mae(y_true,y_pred) / float(scale)

_METRICS = {'mse':mse, 'rmse':rmse, 'mae':mae, 'rmsle':rmsle}

def evaluate_forecasts(results:pd.DataFrame, metrics=('rmse','mae','rmsle'), group_by=None)->pd.DataFrame:
    data=validate_forecast_results(results,group_by=group_by)
    requested=tuple(metrics)
    unknown=set(requested)-set(_METRICS)
    if unknown:
        raise ValueError(f'Unknown metrics: {sorted(unknown)}; use mse/rmse/mae/rmsle')
    cols=[group_by] if isinstance(group_by,str) else list(group_by or [])
    groups = [((),data)] if not cols else data.groupby(cols[0] if len(cols)==1 else cols,sort=False,dropna=False)
    rows=[]
    for key, frame in groups:
        key=(key,) if cols and not isinstance(key,tuple) else key
        row=dict(zip(cols,key)) if cols else {}
        row['n_forecasts']=len(frame)
        for name in requested:
            row[name.upper()]=_METRICS[name](frame['y_true'],frame['y_pred'])
        row['mean_error']=float(np.mean(frame['y_true'].to_numpy()-frame['y_pred'].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)

def interval_summary(results:pd.DataFrame, level=0.95)->pd.DataFrame:
    required=['y_true','interval_lower','interval_upper']
    missing=set(required)-set(results.columns)
    if missing:raise ValueError(f'Missing columns: {sorted(missing)}')
    x=results[required].astype(float)
    if x.isna().any().any():raise ValueError('Intervals and actuals must be observed')
    covered=x['y_true'].between(x['interval_lower'],x['interval_upper'])
    return pd.DataFrame([{'nominal_coverage':level,'empirical_test_coverage':float(covered.mean()),
                          'mean_interval_width_dollars':float((x['interval_upper']-x['interval_lower']).mean()),
                          'n_test_forecasts':len(x)}])

def compare_point_forecasts(results:pd.DataFrame)->pd.DataFrame:
    out=[]
    for label,col in [('Model point forecast','y_pred'),('Validation-bias-adjusted predictive mean','predictive_mean')]:
        out.append({'forecast':label,'test_MSE':mse(results['y_true'],results[col]),
                    'test_RMSE':rmse(results['y_true'],results[col]),
                    'test_MAE':mae(results['y_true'],results[col]),
                    'test_RMSLE':rmsle(results['y_true'],results[col])})
    return pd.DataFrame(out)
