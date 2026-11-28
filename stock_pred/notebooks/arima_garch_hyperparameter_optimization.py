"""ARIMA-GARCH selection using frozen TRAIN/VALIDATION only; no test data.

Run from the project root:
    python stock_pred/notebooks/arima_garch_hyperparameter_optimization.py
Requires: arch pandas numpy.

Grid: AR order of the return mean (0 = random walk with drift in log price,
1 = ARIMA(1,1,0) in log price), GARCH(1,1), normal vs Student-t shocks, and the
rolling window length (GARCH needs longer windows than the 252-session
window used by the other models; 252 often hits the alpha=0 / beta=1 boundary).

Selection criterion: mean one-step QLIKE on validation targets,
    QLIKE = log(sigma^2) + (r - mu)^2 / sigma^2,
the standard loss for comparing volatility forecasts (Patton, 2011);
r is the realized log return, mu and sigma the model's forecast mean and
standard deviation. Point RMSE is reported for reference only.
"""
import argparse
import json
import time
from itertools import product

import numpy as np
import pandas as pd

from optimization_common import frozen_train_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, nargs="+", default=[252, 504, 750])
    parser.add_argument("--ar-orders", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--dists", nargs="+", default=["normal", "t"])
    args = parser.parse_args()

    train, val, root = frozen_train_validation()
    from shared_code.backtest.backtest import univariate_backtest
    from shared_code.models.arima_garch import ARIMAGARCHModel

    if max(args.windows) > len(train):
        raise ValueError(f"Windows must not exceed the {len(train)} training sessions")
    result_dir = root / "stock_pred/results/hyperparameter_optimization"
    result_dir.mkdir(parents=True, exist_ok=True)
    print("TRAIN:", train.index.min().date(), "to", train.index.max().date(), len(train))
    print("VALIDATION:", val.index.min().date(), "to", val.index.max().date(), len(val))
    print("No test CSV or live observations are read.\n")

    series = pd.concat([train, val])
    rows = []
    for window, ar_order, dist in product(args.windows, args.ar_orders, args.dists):
        kwargs = {"ar_order": ar_order, "garch_p": 1, "garch_q": 1, "dist": dist}
        start = time.perf_counter()
        try:
            bt = univariate_backtest(
                series, lambda: ARIMAGARCHModel(**kwargs),
                initial_train_size=window, horizon=1, strategy="rolling",
                window_size=window, start_origin=len(train), end_origin=len(series))
        except Exception as exc:
            print(f"window={window} {kwargs} failed: {str(exc)[:150]}")
            continue
        last = series.shift(1).loc[bt["target_date"]].to_numpy()
        realized = np.log(bt["y_true"].to_numpy() / last)
        mu = np.log(bt["y_pred"].to_numpy() / last)
        sigma = bt["pred_scale"].to_numpy()
        z = (realized - mu) / sigma
        row = {"window_size": window, **kwargs,
               "validation_qlike": float(np.mean(np.log(sigma ** 2) + z ** 2)),
               "validation_rmse": float(np.sqrt(np.mean(bt["error"] ** 2))),
               "z_std": float(np.std(z, ddof=1)),
               "mean_sigma": float(np.mean(sigma)),
               "seconds": round(time.perf_counter() - start, 1)}
        rows.append(row)
        print(f"window={window:3d} AR={ar_order} dist={dist:6s} "
              f"QLIKE={row['validation_qlike']:.4f} RMSE={row['validation_rmse']:.3f} "
              f"std(z)={row['z_std']:.3f}")

    if not rows:
        raise RuntimeError("All ARIMA-GARCH candidates failed.")
    table = pd.DataFrame(rows).sort_values("validation_qlike").reset_index(drop=True)
    table.to_csv(result_dir / "arima_garch_validation.csv", index=False)
    best = table.iloc[0]
    model_kwargs = {"ar_order": int(best.ar_order), "garch_p": 1, "garch_q": 1,
                    "dist": str(best.dist)}
    config = {"model_kwargs": model_kwargs, "window_size": int(best.window_size),
              "validation_qlike": float(best.validation_qlike),
              "validation_rmse": float(best.validation_rmse),
              "selection": "frozen VALIDATION one-step QLIKE; TRAIN/VALIDATION only"}
    (result_dir / "best_arima_garch.json").write_text(json.dumps(config, indent=2))
    print("\nBEST ARIMA-GARCH:", model_kwargs, "| window:", int(best.window_size))
    print("Saved:", result_dir / "best_arima_garch.json")
    print("Copy to tsla_arima_garch.ipynb: MODEL_KWARGS =", model_kwargs,
          "and GARCH_WINDOW_SIZE =", int(best.window_size))
    print("Do not choose a different configuration after examining test-set performance.")


if __name__ == "__main__":
    main()
