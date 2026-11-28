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

### Comparing training windows

Open `kaggle_comp/store_sales.ipynb` for the most recent 365 days or
`kaggle_comp/store_sales_all_history.ipynb` for all available dates. Both
notebooks can read the original CSV files in `kaggle_comp/data/` directly.
Run the cells from top to bottom.

Choose **Fixed predictors** in the first code cell to run Bella's SARIMAX
model. The recent-year notebook compares its July 31 through August 15
holdout forecasts with the saved forecasts from notebook 04. Across all 28,512
holdout rows, the fresh run scored 0.45237696 and Bella’s saved forecasts scored
0.45235291. Both give 0.4524 to four decimal places. The all-history notebook
uses a longer training window, so its score can differ.

Choose **Stepwise selection** to try the extra calendar and promotion
predictors. The search starts with weekdays and compares changes on F1–F3.
The notebook shows the accepted steps. Both options score the final holdout
on every store and family. The Kaggle submission is saved in the same run
folder under `kaggle_comp/results/bella_predictors/`.

### Stepwise results

Both runs used the same eight store–family series on F1–F3. The final holdout
covers all 1,782 series from July 31 through August 15, 2017.

| Training window | Selected predictors | Lists tested | F1–F3 RMSLE on eight series | Final validation RMSLE |
| --- | --- | ---: | ---: | ---: |
| Recent 365 days | Weekdays, promotion, holidays by area, event, trend | 42 | 0.34279 | 0.45327 |
| All history | Weekdays, promotion, holiday | 36 | 0.37916 | 0.46677 |

The day after payday was tested in both runs. The earthquake was tested only
with all history. Neither was selected. In the final holdout, 107 recent-year series and 890
all-history series used the weekday baseline after a failed retry.

An earlier 365-day stepwise run scored 0.51092. One failed fit gave store 43
an extremely high PET SUPPLIES forecast on a local holiday. The stepwise model
now checks whether the fit converged and retries using the training data. On
the same holdout, that forecast was 3.73 units and actual sales were 2. The
corrected score above is a second check on those same dates.
