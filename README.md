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
