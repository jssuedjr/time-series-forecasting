"""Scores and plots shared by the two Store Sales notebooks."""

from pathlib import Path

import numpy as np
import pandas as pd

from shared_code.backtest import kaggle as bt


KEYS = [*bt.K, "date"]


def _check_forecasts(frame):
    if frame.empty or frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Expected one forecast per store, family and date.")
    for column in ["sales", "pred"]:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} must contain finite, nonnegative values.")


def _scores_by(validation, group_column):
    rows = []
    for key, group in validation.groupby(group_column, sort=True, observed=True):
        row = {
            group_column: key,
            "rows": len(group),
            "model_rmsle": bt.rmsle(group.sales, group.pred),
        }
        if "baseline" in group:
            row["baseline_rmsle"] = bt.rmsle(group.sales, group.baseline)
            row["improvement"] = row["baseline_rmsle"] - row["model_rmsle"]
        rows.append(row)
    return pd.DataFrame(rows)


def save_validation(validation, output_dir, model_label, mode, train_days, selection_rmsle=None):
    """Save the forecasts and score every store, family and day together."""
    _check_forecasts(validation)
    expected = pd.date_range(bt.HOLDOUT["HOLDOUT"] + pd.Timedelta(days=1), periods=bt.H)
    dates = pd.DatetimeIndex(validation.date.unique()).sort_values()
    counts = validation.groupby(bt.K, observed=True).size()
    if not np.array_equal(dates, expected) or not counts.eq(bt.H).all():
        raise ValueError("Final validation requires the 16 holdout dates for each series.")
    summary = {
        "training_window": "All available dates" if train_days is None else f"Recent {train_days} days",
        "predictor_choice": mode,
        "predictors": model_label,
        "selection_sample_rmsle": selection_rmsle,
        "final_validation_rmsle": bt.rmsle(validation.sales, validation.pred),
        "final_validation_rows": len(validation),
    }
    if "baseline" in validation:
        summary["weekday_baseline_rmsle"] = bt.rmsle(validation.sales, validation.baseline)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    by_day = _scores_by(validation, "date").rename(columns={"rows": "series"})
    by_day.insert(0, "day", np.arange(1, len(by_day) + 1))
    by_day.to_csv(output_dir / "validation_by_day.csv", index=False)
    _scores_by(validation, "family").to_csv(output_dir / "validation_by_family.csv", index=False)
    validation.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "validation_scores.csv", index=False)
    return summary


def _align_forecasts(current, reference):
    """Match the store, family, date and observed sales before comparing forecasts."""
    for frame in (current, reference):
        _check_forecasts(frame)
    fields = [*KEYS, "sales", "pred"]
    paired = current[fields].merge(
        reference[fields], on=KEYS, how="outer", validate="one_to_one",
        suffixes=("", "_reference"), indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Forecasts cover different store, family or date keys.")
    if not np.array_equal(paired.sales, paired.sales_reference):
        raise ValueError("Observed sales differ for the same store, family and date.")
    return paired


def compare_reference(current, reference_path):
    """Compare a forecast to the saved rows from Bella's notebook."""
    reference = pd.read_parquet(reference_path, columns=[*KEYS, "sales", "pred"])
    paired = _align_forecasts(current, reference)
    current_score = bt.rmsle(paired.sales, paired.pred)
    reference_score = bt.rmsle(paired.sales, paired.pred_reference)
    return {
        "final_validation_rmsle": current_score,
        "bella_reference_rmsle": reference_score,
        "rmsle_difference": current_score - reference_score,
        "matches_at_4_decimals": f"{current_score:.4f}" == f"{reference_score:.4f}",
        "max_prediction_difference": float((paired.pred - paired.pred_reference).abs().max()),
        "rows": len(paired),
    }


def compare_windows(results_dir, mode):
    """Write the overall and per-family scores when both training windows are available."""
    results_dir = Path(results_dir)
    folders = [results_dir / window / mode for window in ("recent_365_days", "all_history")]
    if any(not (folder / "validation_scores.csv").is_file() for folder in folders):
        return pd.DataFrame(), pd.DataFrame()
    scores = pd.concat(
        [pd.read_csv(folder / "validation_scores.csv", float_precision="round_trip") for folder in folders],
        ignore_index=True,
    )
    predictions = [pd.read_parquet(folder / "validation_predictions.parquet",
                                   columns=[*KEYS, "sales", "pred"]) for folder in folders]
    paired = _align_forecasts(*predictions)
    by_family = pd.DataFrame([
        {"family": family, "recent_365_rmsle": bt.rmsle(group.sales, group.pred),
         "all_history_rmsle": bt.rmsle(group.sales, group.pred_reference)}
        for family, group in paired.groupby("family", observed=True, sort=True)
    ])
    by_family["all_minus_recent"] = by_family.all_history_rmsle - by_family.recent_365_rmsle
    scores.to_csv(results_dir / f"training_window_comparison_{mode}.csv", index=False)
    by_family.to_csv(results_dir / f"training_window_by_family_{mode}.csv", index=False)
    return scores, by_family


def _submission_ids(series):
    ids = pd.to_numeric(series, errors="raise")
    if ids.empty or ids.isna().any() or not np.isfinite(ids).all() or ids.mod(1).ne(0).any():
        raise ValueError("Submission IDs must be finite integers.")
    ids = ids.astype("int64")
    if not ids.is_unique:
        raise ValueError("Submission IDs must be unique.")
    return ids


def make_submission(test_forecasts, data_dir, output_dir):
    """Return the Kaggle submission and save it as submission.csv."""
    forecasts = test_forecasts[["id", "pred"]].copy().rename(columns={"pred": "sales"})
    forecasts["id"] = _submission_ids(forecasts.id)
    values = forecasts.sales.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Submission forecasts must be finite and nonnegative.")
    template = Path(data_dir) / "sample_submission.csv"
    if template.is_file():
        ids = _submission_ids(pd.read_csv(template, usecols=["id"]).id)
        if set(ids) != set(forecasts.id):
            raise ValueError("Forecast IDs differ from the sample submission IDs.")
        submission = forecasts.set_index("id").loc[ids].reset_index()
    else:
        submission = forecasts.sort_values("id").reset_index(drop=True)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    submission.to_csv(folder / "submission.csv", index=False)
    return submission


def plot_example(validation, output_dir, store_nbr=44, family="BEVERAGES"):
    """Plot actual sales and forecasts for one series on the RMSLE scale."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    example = validation.loc[validation.store_nbr.eq(store_nbr) & validation.family.eq(family)]
    if example.empty:
        store_nbr, family = validation[bt.K].iloc[0]
        example = validation.loc[validation.store_nbr.eq(store_nbr) & validation.family.eq(family)]
    example = example.sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for column, label, style in [
        ("sales", "Actual sales", "o-"), ("pred", "Forecast", "s-"), ("baseline", "Weekday baseline", "--")
    ]:
        if column in example:
            ax.plot(example.date, np.log1p(example[column]), style, label=label, markersize=4)
    ax.set(title=f"Store {store_nbr} · {family}", xlabel="Validation date", ylabel="log(1 + sales)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend()
    fig.tight_layout()
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / "validation_example.png", dpi=150)
    return fig
