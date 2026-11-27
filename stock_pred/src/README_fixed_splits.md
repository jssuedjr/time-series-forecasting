# TSLA frozen historical experiment

Place these files into the matching locations of your existing project. Your existing model code and the other shared modules remain unchanged.

1. Set the dates in `stock_pred/src/split_config.py` once. The supplied dates reflect the previous 752/251/251 experiment (verify against your local CSV). If you already wrote a plain `split.config`, copy its values into this importable Python file; `split.config` is not importable as a normal Python module.
2. Refresh `stock_pred/data/tsla_historical_data.csv` with completed daily closes.
3. From the project root, run `python -m stock_pred.src.prepare_fixed_splits`. This writes `stock_pred/data/fixed_splits/tsla_train.csv`, `tsla_validation.csv`, and `tsla_test.csv`.
4. Open `stock_pred/notebooks/tsla_rw_test_3_refactored.ipynb` and Run All. The notebook checks that the CSVs match the dates and source history.
5. The notebook exports `stock_pred/results/random_walk_test.csv` and a dated `random_walk_live_<origin>_to_<target>.csv` file.

The live fit uses the most recent 252 observed prices; its calibration uses frozen validation errors. Do not refresh or tune the frozen benchmark using held-out test results. When downloading future data, exclude an incomplete current-session candle. The live snapshot is saved only when an NYSE calendar is available and only if that snapshot file does not already exist.

The frozen CSV inputs contain *observations* for the three periods. The rolling-origin backtest concatenates them so the model may use any past observed price available at the forecast origin. This does not constitute test leakage.
