"""Regression checks for the Store Sales notebook helper; no Kaggle data needed."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from shared_code.backtest import kaggle as bt


@pytest.fixture
def panel():
    dates = pd.date_range('2017-05-01', '2017-08-31')
    index = pd.MultiIndex.from_product(
        [[1, 2], ['GROCERY I', 'BEVERAGES'], dates], names=bt.K + ['date']
    )
    df = index.to_frame(index=False)
    df['family'] = pd.Categorical(df.family, categories=['GROCERY I', 'BEVERAGES', 'UNUSED'])
    df['sales'] = 10.0 + df.date.dt.dayofweek
    df.loc[df.store_nbr == 2, 'sales'] = 0.0
    df.loc[df.date > bt.TEST_CUTOFF, 'sales'] = np.nan
    df['log_sales'] = np.log1p(df.sales)
    df['onpromotion'] = 2
    df['active'] = True
    # Deliberately wrong: the backtester must derive this from its own history.
    df['zero_series'] = df.store_nbr == 1
    df['id'] = np.arange(len(df))
    df['split'] = np.select(
        [df.date > bt.TEST_CUTOFF, df.date > pd.Timestamp('2017-07-30')],
        ['test', 'valid'], default='train'
    )
    return df


def test_parquet_loader_preserves_future_targets_and_ignores_cwd(panel, tmp_path, monkeypatch):
    path = tmp_path / 'processed.parquet'
    panel.sample(frac=1, random_state=12).to_parquet(path, index=False)
    monkeypatch.setattr(bt, 'PROCESSED_PATH', path)
    notebook_dir = tmp_path / 'notebooks'
    notebook_dir.mkdir()
    monkeypatch.chdir(notebook_dir)
    actual = bt.load_panel()
    expected = panel.sort_values(bt.K + ['date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.loc[actual.split == 'test', 'sales'].isna().all()


def test_loader_reports_missing_preprocessing(tmp_path):
    with pytest.raises(FileNotFoundError, match='02_preprocessing.ipynb'):
        bt.load_panel(tmp_path / 'processed.parquet')


def test_loader_rejects_duplicate_keys(panel, tmp_path):
    path = tmp_path / 'processed.parquet'
    pd.concat([panel, panel.iloc[:1]]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match='duplicate'):
        bt.load_panel(path)


def test_random_walk_uses_last_date_including_zero(panel):
    cutoff = bt.FOLDS['F1']
    history = panel.loc[panel.date <= cutoff].copy()
    history.loc[(history.store_nbr == 1) & (history.date == cutoff), 'sales'] = 0
    history = history.sample(frac=1, random_state=9)
    future = panel.loc[panel.date == cutoff + pd.Timedelta(days=1)]
    predictions = bt.random_walk(history, future)
    assert len(predictions) == 4
    assert predictions.pred.eq(0).all()


def test_zero_series_uses_exactly_the_last_30_days(panel):
    cutoff = bt.FOLDS['F1']
    history = panel.loc[panel.date <= cutoff].copy()
    history['sales'] = 0.0
    # Day -30 is outside the window; day -29 is inside it.
    history.loc[(history.store_nbr == 1) & (history.date == cutoff - pd.Timedelta(days=30)), 'sales'] = 100
    history.loc[(history.store_nbr == 2) & (history.date == cutoff - pd.Timedelta(days=29)), 'sales'] = 100
    assert set(bt.zero_series(history)) == {(1, 'GROCERY I'), (1, 'BEVERAGES')}


def test_run_fold_hides_targets_and_recomputes_zero_flags(panel):
    original = panel.copy(deep=True)
    cutoff = bt.FOLDS['F1']

    def forecast(history, future_X):
        assert history.date.max() == cutoff
        assert 'zero_series' not in history
        assert not set(bt.TARGET_COLUMNS).intersection(future_X.columns)
        assert future_X.date.min() == cutoff + pd.Timedelta(days=1)
        assert future_X.date.max() == cutoff + pd.Timedelta(days=bt.H)
        assert 'onpromotion' in future_X
        return future_X[bt.K + ['date']].assign(pred=7.0)

    result = bt.run_fold(forecast, panel, cutoff)
    assert len(result) == 4 * bt.H
    assert result.loc[result.store_nbr == 1, 'pred'].eq(7).all()
    assert result.loc[result.store_nbr == 2, 'pred'].eq(0).all()
    expected_actuals = panel.loc[(panel.date > cutoff) & (panel.date <= cutoff + pd.Timedelta(days=bt.H)), 'sales']
    np.testing.assert_array_equal(result.sales, expected_actuals)
    pd.testing.assert_frame_equal(panel, original)


@pytest.mark.parametrize('bad_output', ['missing', 'duplicate', 'nan'])
def test_run_fold_rejects_invalid_predictions(panel, bad_output):
    def forecast(history, future_X):
        predictions = future_X[bt.K + ['date']].assign(pred=1.0)
        if bad_output == 'missing':
            return predictions.iloc[1:]
        if bad_output == 'duplicate':
            return pd.concat([predictions, predictions.iloc[:1]])
        predictions.iloc[0, predictions.columns.get_loc('pred')] = np.nan
        return predictions

    with pytest.raises(ValueError, match='prediction'):
        bt.run_fold(forecast, panel, bt.FOLDS['F1'])


def test_run_fold_rejects_incomplete_horizon(panel):
    cutoff = bt.FOLDS['F1']
    row = panel.index[(panel.date == cutoff + pd.Timedelta(days=1))][0]
    with pytest.raises(ValueError, match='16 distinct consecutive dates'):
        bt.run_fold(bt.random_walk, panel.drop(index=row), cutoff)


def test_run_fold_rejects_unknown_test_actuals(panel):
    with pytest.raises(ValueError, match='known sales'):
        bt.run_fold(bt.random_walk, panel, bt.TEST_CUTOFF)


def test_rmsle_matches_hand_calculation():
    assert bt.rmsle([0, 3], [0, 7]) == pytest.approx(np.log(2) / np.sqrt(2))
    assert bt.rmsle([0], [-3]) == 0.0
    with pytest.raises(ValueError, match='finite'):
        bt.rmsle([1], [np.nan])


def test_backtest_returns_mean_fold_score(panel):
    result = bt.backtest(bt.random_walk, panel, bt.FOLDS)
    expected = []
    for cutoff in bt.FOLDS.values():
        future = panel.loc[(panel.date > cutoff) & (panel.date <= cutoff + pd.Timedelta(days=16))]
        predicted = np.where(future.store_nbr == 1, 10 + cutoff.dayofweek, 0)
        expected.append(np.sqrt(np.mean((np.log1p(future.sales) - np.log1p(predicted)) ** 2)))
    assert list(result['scores']) == ['F1', 'F2', 'F3']
    assert result['mean'] == pytest.approx(np.mean(expected))


def test_baseline_notebook_runs_from_preprocessing_output(tmp_path):
    """Run the actual notebook cells on a full-size synthetic series grid."""
    source_root = Path(__file__).resolve().parents[1]
    root = tmp_path / 'project'
    notebook_dir = root / 'kaggle_comp' / 'notebooks'
    data_dir = root / 'kaggle_comp' / 'data'
    helper_dir = root / 'shared_code' / 'backtest'
    for directory in [notebook_dir, data_dir, helper_dir]:
        directory.mkdir(parents=True)
    for package in [root / 'shared_code', helper_dir]:
        (package / '__init__.py').write_text('')
    shutil.copy(source_root / 'shared_code/backtest/kaggle.py', helper_dir)
    notebook_name = '03_baseline_random_walk.ipynb'
    shutil.copy(source_root / 'kaggle_comp/notebooks' / notebook_name, notebook_dir)

    families = ['GROCERY I', 'BEVERAGES', 'PRODUCE'] + [f'FAMILY {n}' for n in range(30)]
    index = pd.MultiIndex.from_product(
        [range(1, 55), families, pd.date_range('2017-05-01', '2017-08-31')],
        names=bt.K + ['date']
    )
    df = index.to_frame(index=False)
    df['sales'] = 10.0 + df.store_nbr + df.date.dt.dayofweek
    df.loc[df.family == 'FAMILY 29', 'sales'] = 0.0
    df.loc[df.date > bt.TEST_CUTOFF, 'sales'] = np.nan
    df['log_sales'] = np.log1p(df.sales)
    df['active'] = True
    df['zero_series'] = False
    df['id'] = np.arange(len(df))
    df['split'] = np.select(
        [df.date > bt.TEST_CUTOFF, df.date > pd.Timestamp('2017-07-30')],
        ['test', 'valid'], default='train'
    )
    df['family'] = df.family.astype('category')
    df.to_parquet(data_dir / 'processed.parquet', index=False)

    # New interpreters prevent the test runner's imports from masking path bugs.
    script = '''import json, sys
from pathlib import Path
notebook = json.loads(Path(sys.argv[1]).read_text())
scope = {'__name__': '__main__'}
cells = notebook['cells'] if sys.argv[2] == 'all' else [notebook['cells'][1]]
for cell in cells:
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), 'baseline-notebook', 'exec'), scope)
'''
    environment = dict(os.environ, MPLBACKEND='Agg')
    environment.pop('PYTHONPATH', None)
    for cwd, mode in [(notebook_dir, 'all'), (root, 'setup'), (root / 'kaggle_comp', 'setup')]:
        result = subprocess.run(
            [sys.executable, '-c', script, str(notebook_dir / notebook_name), mode],
            cwd=cwd, env=environment, capture_output=True, text=True, timeout=90
        )
        assert result.returncode == 0, result.stdout + result.stderr

    submission = pd.read_csv(notebook_dir / 'submissions/submission_random_walk.csv')
    assert len(submission) == 28512
    assert submission.id.is_unique and submission.id.is_monotonic_increasing
    assert np.isfinite(submission.sales).all() and submission.sales.ge(0).all()
    last_observations = df.loc[df.date == bt.TEST_CUTOFF, bt.K + ['sales']]
    expected = df.loc[df.split == 'test', bt.K + ['id']].merge(last_observations, on=bt.K)
    expected = expected[['id', 'sales']].sort_values('id').reset_index(drop=True)
    pd.testing.assert_frame_equal(submission, expected)
