"""ARIMA selection: TRAIN-only stationarity/BIC screening; frozen VALIDATION RMSE ranking.
Run: python stock_pred/notebooks/arima_hyperparameter_optimization.py
Requires: pandas numpy scipy statsmodels joblib (optional: pmdarima).
"""
import argparse
import json
import warnings
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from optimization_common import frozen_train_validation, one_step_validation, cache

CACHE_VERSION = 'arima_v1'  # increment after modifying the model or scoring code

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=5, help='BIC-shortlisted orders')
    parser.add_argument('--step', type=int, default=5, help='Validation origin spacing for screening')
    parser.add_argument('--final-step', type=int, default=1, help='Full validation for top candidates')
    parser.add_argument('--window', type=int, default=252)
    parser.add_argument('--max-p', type=int, default=3)
    parser.add_argument('--max-q', type=int, default=3)
    parser.add_argument('--max-d', type=int, default=1)
    parser.add_argument('--use-auto-arima', action='store_true', help='Optional pmdarima candidate')
    args = parser.parse_args()
    train, val, root = frozen_train_validation()
    from shared_code.models.arima import ARIMAModel
    result_dir = root / 'stock_pred/results/hyperparameter_optimization'
    result_dir.mkdir(parents=True, exist_ok=True)
    x = train.to_numpy(dtype=float)
    print('FROZEN train:', train.index.min().date(), 'to', train.index.max().date(), len(train))
    print('FROZEN validation:', val.index.min().date(), 'to', val.index.max().date(), len(val))
    print('No test CSV or live observations are read.\n')
    print('Stationarity tests (TRAIN only; exploratory, not definitive):')
    for label, values in [('price', x), ('first difference', np.diff(x))]:
        try: adf_p = adfuller(values, autolag='AIC')[1]
        except Exception: adf_p = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                kpss_p = kpss(values, regression='c', nlags='auto')[1]
        except Exception: kpss_p = np.nan
        print(f'  {label:20s} ADF p={adf_p:.4g} (H0 unit root); KPSS p={kpss_p:.4g} (H0 stationary)')

    candidates = []
    for p,d,q in product(range(args.max_p+1), range(args.max_d+1), range(args.max_q+1)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit = ARIMA(x, order=(p,d,q), trend=None).fit()
            if np.isfinite(fit.bic): candidates.append({'order':(p,d,q), 'bic':float(fit.bic)})
        except Exception as e:
            print(f'Skipped ARIMA{(p,d,q)} in BIC screening: {str(e)[:100]}')
    if args.use_auto_arima:
        try:
            from pmdarima import auto_arima
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                selected = auto_arima(x, seasonal=False, start_p=0, start_q=0,
                                      max_p=args.max_p, max_q=args.max_q, max_d=args.max_d,
                                      test='adf', information_criterion='bic', stepwise=True,
                                      suppress_warnings=True, error_action='ignore')
            print('pmdarima suggested:', selected.order)
            if selected.order not in [r['order'] for r in candidates]:
                candidates.append({'order':tuple(selected.order), 'bic':float(selected.bic())})
        except Exception as e:
            print('Optional pmdarima unavailable:', e)
    if not candidates: raise RuntimeError('No ARIMA orders fitted successfully.')
    bic_table = pd.DataFrame(candidates).sort_values('bic').reset_index(drop=True)
    bic_table.assign(order=bic_table.order.astype(str)).to_csv(result_dir/'arima_train_bic.csv',index=False)
    shortlist = bic_table.head(args.top)['order'].tolist()
    print('\nTop TRAIN-only BIC orders:')
    print(bic_table.head(args.top).to_string(index=False))

    memory = cache(root)
    @memory.cache
    def evaluate_arima_order(order, train_values, val_values, window_size, step, version):
        # version explicitly participates in cache key
        return one_step_validation(ARIMAModel, {'order':tuple(order), 'trend':None},
                pd.Series(train_values), pd.Series(val_values), window_size=window_size, step=step)[0]

    coarse = []
    for order in shortlist:
        try:
            rmse = evaluate_arima_order(tuple(order), train.to_numpy(), val.to_numpy(),
                                        args.window, args.step, CACHE_VERSION)
            coarse.append({'order':tuple(order), 'validation_rmse_step':rmse})
            print(f'ARIMA{tuple(order)} step={args.step}: validation RMSE={rmse:.5f}')
        except Exception as e:
            print('Validation failed for', order, str(e)[:150])
    if not coarse: raise RuntimeError('All validation candidates failed.')
    coarse_df = pd.DataFrame(coarse).sort_values('validation_rmse_step')
    coarse_df.assign(order=coarse_df.order.astype(str)).to_csv(result_dir/'arima_validation_screen.csv',index=False)
    # Refine the top 3 candidates using every validation date, then choose the winner.
    fine = []
    for order in coarse_df.head(3)['order']:
        try:
            rmse = evaluate_arima_order(tuple(order), train.to_numpy(), val.to_numpy(),
                                        args.window, args.final_step, CACHE_VERSION)
            fine.append({'order':tuple(order), 'validation_rmse':rmse})
            print(f'ARIMA{tuple(order)} final step={args.final_step}: validation RMSE={rmse:.5f}')
        except Exception as e:
            print('Full validation failed for', order, str(e)[:150])
    if not fine: raise RuntimeError('All final validation candidates failed.')
    final = pd.DataFrame(fine).sort_values('validation_rmse').reset_index(drop=True)
    final.assign(order=final.order.astype(str)).to_csv(result_dir/'arima_validation_final.csv',index=False)
    best = tuple(final.iloc[0]['order'])
    config = {'order':list(best), 'trend':None, 'validation_rmse':float(final.iloc[0]['validation_rmse']),
              'selection':'TRAIN-only BIC screen + frozen VALIDATION RMSE',
              'window_size':args.window, 'validation_step':args.final_step}
    (result_dir/'best_arima.json').write_text(json.dumps(config, indent=2))
    print('\nBEST ARIMA:', best, '| validation RMSE:',config['validation_rmse'])
    print('Saved:', result_dir/'best_arima.json')
    print('Copy to tsla_arima.ipynb: MODEL_KWARGS =', {'order':best, 'trend':None})
    print('Do not choose a different order after examining test-set performance.')

if __name__=='__main__': main()
