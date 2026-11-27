"""Frozen-split, one-step-ahead validation utilities. No test observations are loaded."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from joblib import Memory

def project_root():
    cwd = Path.cwd().resolve()
    root = next((p for p in (cwd, *cwd.parents)
                 if (p / "pyproject.toml").exists() and (p / "shared_code").is_dir()), None)
    if root is None:
        raise RuntimeError("Run from within your timeseriesproj repository.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root

def frozen_train_validation():
    root = project_root()
    folder = root / "stock_pred/data/fixed_splits"
    def read(name):
        df = pd.read_csv(folder / f"tsla_{name}.csv", parse_dates=["Date"])
        s = df.set_index("Date")["Close"].astype(float).sort_index()
        if s.index.has_duplicates or not np.isfinite(s.to_numpy()).all():
            raise ValueError(f"Invalid frozen {name} data")
        return s
    train, val = read("train"), read("validation")
    if not train.index.max() < val.index.min():
        raise ValueError("Training and validation dates overlap.")
    return train, val, root

def cache(root):
    return Memory(location=root / "stock_pred/results/cache/hyperparameters", verbose=0)

def one_step_validation(model_class, config, train, val, *, window_size=252, step=5):
    """Refit using only observations before each target; score only validation targets."""
    if step < 1 or window_size < 2:
        raise ValueError("step and window_size must be positive.")
    series = pd.concat([train, val])
    values = series.to_numpy(dtype=float)
    ntrain = len(train)
    rows = []
    for target in range(ntrain, len(values), step):
        window = values[max(0, target-window_size):target]
        model = model_class(**config)
        model.fit(window)
        pred = float(np.asarray(model.predict(1), dtype=float).reshape(-1)[0])
        if not np.isfinite(pred):
            raise ValueError(f"Non-finite prediction for {series.index[target]}")
        rows.append((series.index[target], values[target], pred))
    frame = pd.DataFrame(rows, columns=["target_date", "y_true", "y_pred"])
    frame["error"] = frame["y_true"] - frame["y_pred"]
    rmse = float(np.sqrt(np.mean(frame["error"].to_numpy() ** 2)))
    return rmse, frame

