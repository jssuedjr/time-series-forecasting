"""Optuna MLP tuning using frozen TRAIN/VALIDATION only; no test data.
Run: python stock_pred/notebooks/mlp_parameter_optimization.py --trials 25
Requires: optuna joblib pandas numpy and a WORKING shared_code.models.mlp.MLPModel.
"""
import argparse
import json
import numpy as np
import pandas as pd
from optimization_common import frozen_train_validation, one_step_validation, cache

CACHE_VERSION = 'mlp_v1'  # bump if model code or evaluation changes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--final-step', type=int, default=1)
    parser.add_argument('--window', type=int, default=252)
    args = parser.parse_args()
    try: import optuna
    except ImportError as e: raise SystemExit('Install with: pip install optuna joblib') from e
    train, val, root = frozen_train_validation()
    from shared_code.models.mlp import MLPModel
    result_dir = root/'stock_pred/results/hyperparameter_optimization'
    result_dir.mkdir(parents=True, exist_ok=True)
    print('TRAIN:',train.index.min().date(),'to',train.index.max().date())
    print('VALIDATION:',val.index.min().date(),'to',val.index.max().date())
    print('No test CSV or live observations are read.')
    memory = cache(root)
    @memory.cache
    def evaluate_mlp(config_tuple, train_values, val_values, window_size, step, version):
        config = dict(config_tuple)
        return one_step_validation(MLPModel, config, pd.Series(train_values),
             pd.Series(val_values), window_size=window_size, step=step)[0]

    def config_from_params(params, seed=42):
        return dict(n_lags=int(params['n_lags']), hidden_size=int(params['hidden_size']),
                    epochs=int(params['epochs']), learning_rate=float(params['learning_rate']),
                    random_state=seed)
    def score(config, step):
        return evaluate_mlp(tuple(sorted(config.items())), train.to_numpy(), val.to_numpy(),
                            args.window, step, CACHE_VERSION)
    def objective(trial):
        params = dict(n_lags=trial.suggest_categorical('n_lags',[5,10,20,30]),
                      hidden_size=trial.suggest_categorical('hidden_size',[8,16,32]),
                      epochs=trial.suggest_categorical('epochs',[25,50,100]),
                      learning_rate=trial.suggest_float('learning_rate',1e-4,1e-2,log=True))
        config = config_from_params(params)
        try:
            rmse = score(config,args.step)
            if not np.isfinite(rmse): raise ValueError('non-finite RMSE')
            return rmse
        except Exception as e:
            print('Trial failed:',e)
            raise optuna.TrialPruned()
    db = result_dir/'mlp_optuna.sqlite3'
    study = optuna.create_study(direction='minimize', study_name='tsla_mlp_v1',
                                storage=f'sqlite:///{db.resolve()}', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,n_trials=args.trials)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed: raise RuntimeError('No successful MLP trials. Check MLPModel.fit/predict implementation.')
    completed.sort(key=lambda t:t.value)
    # Full daily validation for best three configurations; same validation targets.
    fine = []
    for trial in completed[:3]:
        config=config_from_params(trial.params)
        try:
            rmse=score(config,args.final_step)
            fine.append((rmse,config))
            print('Full validation RMSE:',round(rmse,5),'params:',config)
        except Exception as e:
            print('Full validation failed:',e)
    if not fine: raise RuntimeError('No final candidate succeeded.')
    fine.sort(key=lambda x:x[0]); best_rmse, best=fine[0]
    # Seed sensitivity of winning architecture (same frozen validation, no test).
    seed_scores=[]
    for seed in (42,123,2026):
        config={**best,'random_state':seed}
        seed_scores.append({'seed':seed,'validation_rmse':score(config,args.final_step)})
    pd.DataFrame(seed_scores).to_csv(result_dir/'mlp_seed_stability.csv',index=False)
    config={'model_kwargs':best,'validation_rmse':float(best_rmse),
            'seed_validation_rmse':seed_scores,
            'selection':'frozen VALIDATION one-step RMSE; train window only; Optuna search',
            'window_size':args.window,'validation_step':args.final_step}
    (result_dir/'best_mlp.json').write_text(json.dumps(config,indent=2))
    print('\nBEST MLP:',best,'| validation RMSE:',best_rmse)
    print('Saved:',result_dir/'best_mlp.json')
    print('Copy to tsla_nn.ipynb: MODEL_KWARGS =',best)
    print('Do not tune using test results.')

if __name__=='__main__':main()
