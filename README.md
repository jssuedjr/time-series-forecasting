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

Run `kaggle_comp/store_sales.ipynb` for the most recent 365 days or `kaggle_comp/store_sales_all_history.ipynb` for all available dates. Choose **Fixed predictors** or **Stepwise selection** in the first code cell. These two notebooks read the original CSV files in `kaggle_comp/data/` directly. Run the cells from top to bottom.

Both choices use `AutoARIMA(season_length=7, ic="bic")` from StatsForecast 2.0.3. AutoARIMA uses BIC to choose the error model for each store and product family.

| Training window | Predictor choice | Predictors | F1–F3 RMSLE on eight series | Final validation RMSLE |
| --- | --- | --- | ---: | ---: |
| Recent 365 days | Fixed predictors | `log_promo`, `holiday`, `event`, `closed` | — | 0.45238 |
| Recent 365 days | Stepwise selection | Weekdays, promotion, holiday by area, trend | 0.33849 | 0.44538 |
| All history | Stepwise selection | Weekdays, promotion | 0.35521 | 0.44713 |
| All history | Fixed predictors | `log_promo`, `holiday`, `event`, `closed` | — | Not rerun |

**Fixed predictors** uses Bella’s four predictors directly. The recent-year run scored 0.45237696. Bella’s saved forecasts scored 0.45235291 on the same 28,512 rows. Both round to **0.4524**, although individual predictions differ. No predictor search is needed for this choice.

**Stepwise selection** starts with weekdays and compares predictor groups on F1–F3. Candidates include the day after payday, holiday and event types, and previous-day promotion. The earthquake flag can be tested with all history because that training window includes April 2016. The recent-year model chose separate national, regional and local holiday indicators. The original `observerhan` branch uses Best subset with 48 predictor lists from a smaller set of candidates.

The two branches score predictor selection on the same eight store–family series. The final validation RMSLE uses all 1,782 series over July 31–August 15, 2017, giving 28,512 forecasts. Lower RMSLE is better. Earlier Stepwise selection results in `kaggle_comp/results/bella_predictors/` used restricted AutoARIMA settings.

The recent-year search tested 43 predictor lists, and the all-history search tested 25. Stepwise selection keeps a change when it lowers the sampled F1–F3 RMSLE by at least 0.0005. With all history, adding a holiday or make-up workday flag improved the score by less than this amount, so selection stopped after promotion.

The day after payday was tested in both windows. The earthquake flag was tested with all history. Neither was selected.

The score tables, accepted steps and candidate scores are in [the saved results](kaggle_comp/results/bella_settings/). The notebooks also save the full forecasts when run.

The recent-year stepwise holdout had 2 fits with rank warnings, and the all-history holdout had 69. Each of these fits selected weekly differencing, which cancels the repeating weekday indicators. Their separate weekday effects cannot be estimated uniquely. The forecasts were finite and included in the scores. Neither run used the weekday fallback. The notebooks save these notes in `validation_fits.csv`.
