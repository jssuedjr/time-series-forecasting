import numpy as np
import pandas as pd
import pytest

from kaggle_comp.src.results import compare_reference, compare_windows, make_submission, save_validation
from shared_code.backtest import kaggle as bt


@pytest.fixture
def validation():
    dates = pd.date_range("2017-07-31", periods=16)
    sales = np.tile(np.arange(16, dtype=float), 2)
    return pd.DataFrame({
        "date": np.tile(dates, 2),
        "store_nbr": 1,
        "family": ["BREAD"] * 16 + ["EGGS"] * 16,
        "sales": sales,
        "pred": np.r_[sales[:16], sales[16:] + 2],
    })


def test_reference_compares_same_rows_and_sales(validation, tmp_path):
    path = tmp_path / "reference.parquet"
    validation.sample(frac=1, random_state=42).to_parquet(path, index=False)
    check = compare_reference(validation, path)
    assert check["matches_at_4_decimals"]
    assert check["max_prediction_difference"] == 0
    changed = validation.copy()
    changed.loc[0, "sales"] += 1
    with pytest.raises(ValueError, match="Observed sales differ"):
        compare_reference(changed, path)
    with pytest.raises(ValueError, match="different store, family or date"):
        compare_reference(validation.iloc[1:], path)
    changed = validation.copy()
    changed.loc[0, "pred"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compare_reference(changed, path)


def test_window_comparison_waits_for_same_mode(validation, tmp_path):
    for window, mode, days in [
        ("recent_365_days", "fixed", 365), ("all_history", "best_subset", None)
    ]:
        save_validation(validation, tmp_path / window / mode, "run", mode, days)
    empty, _ = compare_windows(tmp_path, "fixed")
    assert empty.empty
    improved = validation.assign(pred=validation.sales)
    save_validation(improved, tmp_path / "all_history" / "fixed", "run", "fixed", None)
    scores, families = compare_windows(tmp_path, "fixed")
    assert np.allclose(scores.final_validation_rmsle, [bt.rmsle(validation.sales, validation.pred), 0])
    assert list(scores.final_validation_rows) == [32, 32]
    assert families.set_index("family").loc["EGGS", "all_minus_recent"] < 0
    with pytest.raises(ValueError, match="16 holdout dates"):
        save_validation(validation.iloc[1:], tmp_path / "other", "run", "fixed", 365)


def test_submission_preserves_template_order_and_checks_coverage(tmp_path):
    forecasts = pd.DataFrame({"id": [5., 3.], "pred": [7., 10.]})
    result = make_submission(forecasts, tmp_path, tmp_path)
    assert result.id.tolist() == [3, 5]
    pd.DataFrame({"id": [5, 3], "sales": [0, 0]}).to_csv(tmp_path / "sample_submission.csv", index=False)
    result = make_submission(forecasts.iloc[::-1], tmp_path, tmp_path)
    assert list(result.itertuples(index=False, name=None)) == [(5, 7.), (3, 10.)]
    with pytest.raises(ValueError, match="differ"):
        make_submission(forecasts.iloc[:1], tmp_path, tmp_path)
    with pytest.raises(ValueError, match="integers"):
        make_submission(forecasts.assign(id=[5.3, 3.]), tmp_path, tmp_path)
