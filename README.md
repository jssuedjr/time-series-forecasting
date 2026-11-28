Create a virtual environment and install the project from the repository root.

Follow these steps if running for the first time:
In terminal, run from repository root: 

"
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python -m pip install -e ".[dev]"

python -m ipykernel install --user \
    --name ams586 \
    --display-name "Python (AMS 586)"
"

The shared_code folder becomes a Python project to avoid pathing issues when using functions from shared_code for each sub-problem.

## Kaggle notebooks

Place the competition CSV files in `kaggle_comp/data/`, then run
`02_preprocessing.ipynb` from `kaggle_comp/notebooks/`. It saves
`kaggle_comp/data/processed.parquet`; the optional CSV export is for inspection.

Notebooks 03 and 04 use `shared_code/backtest/kaggle.py`. Their setup cells find
the repository root, and the loader reads the Parquet file independently of the
kernel's working directory. Run the setup cell again after restarting a kernel.
If the Parquet file is missing, the loader asks you to run notebook 02 first.

The Parquet dependency (`pyarrow`) is included in the project dependencies and
`requirements.txt`. Notebook 04 also needs StatsForecast. To install the project,
notebook tools, tests, and SARIMAX dependencies together, run from the repository root:

```bash
python -m pip install -e ".[dev,kaggle]"
```

The Kaggle helper preserves the documented F1–F3 and holdout dates. It withholds
future sales, log sales, and sales-derived flags from forecasting functions and
recomputes the 30-day zero-sales rule from each fold's history. The existing
univariate stock backtester remains in `shared_code/backtest/backtest.py`.

Run the tests with `python -m pytest`. Forecasting tests use synthetic panels;
they do not download competition data or establish Kaggle accuracy scores.

### Store sales results using Bella’s AutoARIMA settings

Run `kaggle_comp/store_sales.ipynb` for the most recent 365 days or `kaggle_comp/store_sales_all_history.ipynb` for all available dates. These two notebooks read the original CSV files in `kaggle_comp/data/` directly. Run the cells from top to bottom.

Both use `AutoARIMA(season_length=7, ic="bic")` from StatsForecast 2.0.3 for this comparison. AutoARIMA uses BIC to choose the error model for each store and product family.

**Best subset** tests all 48 predictor lists in these notebooks. Each list includes weekday indicators. The optional predictors are promotion, holiday, payday, trend and up to two annual Fourier pairs. Selection uses the RMSLE across F1–F3 for the same eight store–family series in both branches. The final validation RMSLE uses all 1,782 series over July 31–August 15, 2017, giving 28,512 forecasts. Lower RMSLE is better.

| Training window | Selected predictors | F1–F3 RMSLE on eight series | Final validation RMSLE |
| --- | --- | ---: | ---: |
| Recent 365 days | Weekdays, promotion, holiday, trend | 0.33899 | 0.44489 |
| All history | Weekdays, promotion, holiday | 0.35515 | 0.44652 |

Both selected models use zero annual Fourier pairs (`K=0`). The optional fixed-predictor comparison with weekdays, promotion and two annual Fourier pairs was not rerun. The earlier outputs in `kaggle_comp/results/store_sales_365d/` and `kaggle_comp/results/store_sales_all_history/` used restricted AutoARIMA settings.

The score tables and candidate scores are in [the recent 365-day results](kaggle_comp/results/store_sales_365d_bella_settings/) and [the all-history results](kaggle_comp/results/store_sales_all_history_bella_settings/). The earlier results remain in their original folders.

The recent-year holdout had 2 fits with rank warnings, and the all-history holdout had 67. Each of these fits selected weekly differencing, which cancels the repeating weekday indicators. Their separate weekday effects cannot be estimated uniquely. The forecasts were finite and included in the scores. Neither run used the weekday fallback. The notebooks save these notes in `validation_fits.csv`.
