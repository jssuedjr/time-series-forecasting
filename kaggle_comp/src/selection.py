"""Select predictors on F1–F3 and forecast with the selected model."""

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from shared_code.backtest import kaggle as bt


WEEKDAYS = [f"dow_{day}" for day in range(1, 7)]
FOURIER = ["sin1", "cos1", "sin2", "cos2"]


def baseline_one(history, future):
    """Average log sales on the same weekday in the last four weeks."""
    recent = history.loc[history.date > history.date.max() - pd.Timedelta(days=28)]
    recent = recent.dropna(subset=["sales"])
    if recent.empty:
        return np.zeros(len(future))
    logs = np.log1p(recent.sales)
    weekday_means = logs.groupby(recent.date.dt.weekday).mean()
    forecast = future.date.dt.weekday.map(weekday_means).fillna(logs.mean())
    return np.expm1(forecast.to_numpy(dtype=float))


def usable_columns(history, columns):
    """Drop constant or redundant predictors using this series' training dates."""
    keep = []
    design = np.ones((len(history), 1))
    for column in columns:
        if column in FOURIER and len(history) < 365:
            continue
        values = history[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Training predictor {column} has missing or infinite values.")
        if len(values) < 2 or values.min() == values.max():
            continue
        trial = np.column_stack((design, values))
        if np.linalg.matrix_rank(trial) == trial.shape[1]:
            keep.append(column)
            design = trial
    return keep


def _training_log_sales(history):
    # Fill missing Christmas rows from earlier observations on the same weekday.
    y = np.log1p(history.sales.to_numpy(dtype=float))
    for i in np.flatnonzero(np.isnan(y)):
        matching = y[np.arange(i - 7, max(-1, i - 29), -7)]
        matching = matching[np.isfinite(matching)]
        if len(matching):
            y[i] = matching.mean()
        else:
            earlier = y[:i][np.isfinite(y[:i])]
            y[i] = earlier.mean() if len(earlier) else 0.0
    return y


def _retry_fit(selected, y, x_train):
    """Retry the selected orders with direct maximum likelihood."""
    from statsforecast.arima import Arima

    p, q, P, Q, period, d, D = selected["arma"]
    coef = selected["coef"]
    # The low-level fit accepts SciPy BFGS's standard gradient tolerance.
    return Arima(y, order=(p, d, q), seasonal={"order": (P, D, Q), "period": period},
                 xreg=x_train, include_mean=("intercept" in coef or "constant" in coef),
                 include_drift="drift" in coef, method="ML",
                 optim_control={"maxiter": 100, "gtol": 1e-5})


def forecast_one(history, future, columns):
    """Fit the weekday regression with ARIMA errors for one store and family."""
    from statsforecast.arima import arima_string
    from statsforecast.models import AutoARIMA

    observed = history.sales.dropna()
    info = {
        "method": "", "order": "", "bic": np.nan, "days": len(history),
        "history_start": history.date.min(), "history_end": history.date.max(),
        "used_columns": "", "note": "", "acf_lag1": np.nan, "acf_lag7": np.nan,
    }
    if observed.empty or observed.nunique() == 1:
        value = 0.0 if observed.empty else float(observed.iloc[-1])
        info.update(method="constant", note="No observed sales" if observed.empty else "Constant sales")
        return np.full(len(future), value), info
    if len(observed) < 35:
        info.update(method="weekday baseline", note="Fewer than five weeks of observed sales")
        return baseline_one(history, future), info

    keep = usable_columns(history, columns)
    info["used_columns"] = ", ".join(keep)
    x_train = history[keep].to_numpy(dtype=float) if keep else None
    x_future = future[keep].to_numpy(dtype=float) if keep else None
    if keep and not np.isfinite(x_future).all():
        raise ValueError("A future predictor has missing or infinite values.")
    y = _training_log_sales(history)
    # Weekly differencing would remove the weekday indicators.
    model = AutoARIMA(
        season_length=7, D=0, max_d=1, allowdrift=False, ic="bic",
        max_p=1, max_q=1, max_P=1, max_Q=1,
        start_p=0, start_q=0, start_P=0, start_Q=0,
        stepwise=True, nmodels=12, approximation=True,
    )
    notes = []
    try:
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            model.fit(y, X=x_train)
            code = model.model_["code"]
            if code != 0:
                notes.append(f"Final fit returned code {code}. The model retried the same orders using ML.")
                fit = _retry_fit(model.model_, y, x_train)
                if fit["code"] != 0:
                    raise ValueError(f"ML retry returned code {fit['code']}.")
                if not np.isfinite([fit["bic"], fit["sigma2"], *fit["coef"].values()]).all():
                    raise ValueError("ML retry returned nonfinite estimates.")
                model.model_ = fit
            log_prediction = model.predict(h=len(future), X=x_future)["mean"]
            prediction = np.expm1(np.maximum(0, np.asarray(log_prediction)))
        if prediction.shape != (len(future),) or not np.isfinite(prediction).all():
            raise ValueError("Invalid forecast")
    except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError) as error:
        notes.append(f"{type(error).__name__}: {error}")
        info.update(method="weekday baseline", note=" ".join(notes))
        return baseline_one(history, future), info

    errors = pd.Series(model.model_["residuals"])
    info.update(method="AutoARIMA", order=arima_string(model.model_).strip(),
                bic=float(model.model_["bic"]), acf_lag1=errors.autocorr(1),
                acf_lag7=errors.autocorr(7),
                note=" | ".join(dict.fromkeys([*notes, *(str(w.message) for w in fit_warnings)])))
    return prediction, info


# Keep each annual sine/cosine pair and each set of holiday or event types together.
PREDICTOR_GROUPS = {
    "promotion": ["log_promo"],
    "previous promotion": ["log_promo_lag1"],
    "holiday": ["holiday"],
    "holiday by area": ["hol_national", "hol_regional", "hol_local"],
    "holiday neighbors": ["before_holiday", "after_holiday"],
    "event": ["event"],
    "event types": ["event_shopping", "event_sports", "event_other"],
    "closed": ["closed"],
    "make-up workday": ["work_day"],
    "payday": ["payday"],
    "day after payday": ["after_payday"],
    "day after payday on a weekend": ["after_payday_weekend"],
    "earthquake": ["earthquake"],
    "trend": ["trend_years"],
    "annual pair 1": FOURIER[:2],
    "annual pair 2": FOURIER[2:],
}


def _valid_groups(state):
    return (
        ("previous promotion" not in state or "promotion" in state)
        and ("day after payday" not in state or "payday" in state)
        and ("day after payday on a weekend" not in state or "day after payday" in state)
        and ("annual pair 2" not in state or "annual pair 1" in state)
        and ("holiday neighbors" not in state or bool(state & {"holiday", "holiday by area"}))
    )


def _neighbors(state, groups, allow_removal):
    """Propose one change, keeping related predictors together."""
    replacements = {"holiday": "holiday by area", "holiday by area": "holiday",
                    "event": "event types", "event types": "event"}
    choices = []
    for name in groups:
        if name in state:
            if not allow_removal:
                continue
            changed = state - {name}
        else:
            changed = (state - {replacements.get(name)}) | {name}
            # Test the delayed response even if payday alone does not improve the score.
            if name == "day after payday" and "payday" in groups:
                changed = changed | {"payday"}
        if _valid_groups(changed):
            choices.append(changed)
    return choices


def _stepwise(groups, evaluate, max_steps, min_improvement, allow_removal):
    """Score each list once. Return all tried lists and the accepted path."""
    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer.")
    if not np.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError("min_improvement must be finite and nonnegative.")
    candidates, scores = {}, {}

    def label(state):
        return " + ".join(["weekdays"] + [g for g in groups if g in state])

    def assess(states):
        new = {}
        for state in states:
            name = label(state)
            if name not in scores:
                new[name] = WEEKDAYS + [c for g, cols in groups.items() if g in state for c in cols]
        if new:
            scores.update(evaluate(new))
            candidates.update(new)

    state = frozenset()
    assess([state])
    path = [label(state)]
    for step in range(1, max_steps + 1):
        choices = _neighbors(state, groups, allow_removal)
        if not choices:
            break
        assess(choices)
        # On an exact tie, prefer fewer predictors and then the alphabetical label.
        best = min(choices, key=lambda s: (scores[label(s)], len(candidates[label(s)]), label(s)))
        improvement = scores[label(state)] - scores[label(best)]
        if improvement <= 1e-12 or improvement + 1e-12 < min_improvement:
            break
        state = best
        path.append(label(state))
        print(f"Step {step}: {path[-1]} (F1–F3 RMSLE {scores[path[-1]]:.5f})")
    return candidates, path


def sample_series(panel, series_per_group):
    """Use the last 90 days through F1's cutoff to draw the same sample for every model."""
    cutoff = min(bt.FOLDS.values())
    recent = panel.loc[panel.date.between(cutoff - pd.Timedelta(days=89), cutoff),
                       [*bt.K, "observed", "sales"]]
    values = np.log1p(recent.sales.where(recent.observed))
    levels = values.groupby([recent.store_nbr, recent.family], observed=True).mean()
    levels = levels.sort_index().rename("mean_log_sales").reset_index()
    if series_per_group is None:
        return levels
    if series_per_group < 1:
        raise ValueError("series_per_group must be positive, or None for all series.")
    active = levels.loc[levels.mean_log_sales.gt(0)].copy()
    if active.empty:
        raise ValueError("No active series were found before F1.")
    groups = min(4, len(active))
    active["sales_group"] = pd.qcut(active.mean_log_sales.rank(method="first"), groups, labels=False)
    grouped = active.groupby("sales_group", group_keys=False)
    if grouped.size().min() >= series_per_group:
        return grouped.sample(n=series_per_group, random_state=586)
    return pd.concat([g.sample(n=min(len(g), series_per_group), random_state=586) for _, g in grouped])


def _split_series(frame, cutoff, columns, train_days):
    before = frame.date.le(cutoff) & frame.store_started
    if train_days is not None:
        before &= frame.date.gt(cutoff - pd.Timedelta(days=train_days))
    history = frame.loc[before, ["date", "sales", *columns]].copy()
    history["sales"] = history.sales.where(frame.loc[before, "observed"])
    end = cutoff + pd.Timedelta(days=bt.H)
    rows = frame.loc[frame.date.gt(cutoff) & frame.date.le(end)]
    expected = pd.date_range(cutoff + pd.Timedelta(days=1), periods=bt.H)
    if not np.array_equal(rows.date.to_numpy(), expected.to_numpy()):
        raise ValueError(f"Expected {bt.H} consecutive forecast dates after {cutoff:%Y-%m-%d}.")
    return history, rows, rows[["date", *columns]]


def _score_series(key, frame, candidates, train_days):
    results = []
    columns = list(dict.fromkeys(c for cols in candidates.values() for c in cols))
    for period, cutoff in bt.FOLDS.items():
        history, rows, future = _split_series(frame, cutoff, columns, train_days)
        # Reuse a fit when two lists differ only by unavailable predictors for this series.
        varying = set(history[columns].nunique(dropna=False).loc[lambda counts: counts > 1].index)
        if len(history) < 365:
            varying.difference_update(FOURIER)
        cache = {}
        for name, cols in candidates.items():
            active = tuple(c for c in cols if c in varying)
            if active not in cache:
                pred, info = forecast_one(history, future, active)
                cache[active] = (bt.rmsle(rows.sales, pred), info)
            score, info = cache[active]
            results.append(dict(store_nbr=key[0], family=key[1], candidate=name, period=period,
                                rmsle=score, method=info["method"], order=info["order"],
                                used_columns=info["used_columns"]))
    return results


def _rank(details, candidates):
    ranking = (details.assign(msle=details.rmsle**2).groupby("candidate").msle.mean()
               .pow(0.5).rename("selection_rmsle").to_frame())
    ranking["predictors"] = [len(candidates[name]) for name in ranking.index]
    ranking["columns"] = [", ".join(candidates[name]) for name in ranking.index]
    ranking = ranking.join(pd.crosstab(details.candidate, details.method))
    return ranking.sort_values(["selection_rmsle", "predictors"], kind="stable")


def _frames(panel, columns):
    fields = [*bt.K, "date", "id", "sales", "observed", "store_started", *columns]
    for key, frame in panel[fields].groupby(bt.K, observed=True, sort=True):
        yield key, frame.sort_values("date").reset_index(drop=True)


def select_predictors(panel, train_days, series_per_group=2, n_jobs=4, *,
                      max_steps=4, min_improvement=0.0005, allow_removal=True):
    """Run a stepwise search on F1–F3. Return winner, columns, sample, details, and ranking."""
    sample = sample_series(panel, series_per_group)
    if series_per_group is None:
        source = panel
    else:
        by_series = panel.groupby(bt.K, observed=True, sort=False).indices
        rows = [by_series[key] for key in sample[bt.K].itertuples(index=False, name=None)]
        source = panel.iloc[np.concatenate(rows)]
    # F3 ends on July 30. Availability uses training dates only.
    development = source.loc[source.date.le(max(bt.FOLDS.values()) + pd.Timedelta(days=bt.H))]
    training = development.date.le(max(bt.FOLDS.values())) & development.store_started
    if train_days is not None:
        training &= development.date.gt(min(bt.FOLDS.values()) - pd.Timedelta(days=train_days))
    groups = {}
    for name, columns in PREDICTOR_GROUPS.items():
        available = [c for c in columns
                     if c in development and development.loc[training, c].nunique() > 1]
        if available:
            groups[name] = available

    batches = []
    def evaluate(lists):
        cols = list(dict.fromkeys(c for candidate in lists.values() for c in candidate))
        with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
            per_series = Parallel()(delayed(_score_series)(key, frame, lists, train_days)
                                    for key, frame in _frames(development, cols))
        details = pd.DataFrame([r for series_scores in per_series for r in series_scores])
        batches.append(details)
        return _rank(details, lists)["selection_rmsle"].to_dict()

    candidates, path = _stepwise(groups, evaluate, max_steps, min_improvement, allow_removal)
    details = pd.concat(batches, ignore_index=True)
    if not details.method.eq("AutoARIMA").any():
        raise RuntimeError("No AutoARIMA fit succeeded on the selection series.")
    ranking = _rank(details, candidates)
    steps = {label: step for step, label in enumerate(path)}
    ranking["selection_step"] = pd.array(ranking.index.map(steps), dtype="Int64")
    winner = path[-1]
    ranking["selected"] = ranking.index == winner
    return winner, candidates[winner], sample, details, ranking


def _predict_series(key, frame, cutoff, columns, train_days):
    history, rows, future = _split_series(frame, cutoff, columns, train_days)
    prediction, info = forecast_one(history, future, columns)
    result = rows[["id", "date", "sales"]].copy()
    result["store_nbr"], result["family"] = key
    result["pred"] = prediction
    result["baseline"] = baseline_one(history, future)
    return result, dict(store_nbr=key[0], family=key[1], **info)


def forecast_selected(panel, cutoff, columns, train_days, n_jobs=4):
    """Refit the selected predictor list for all series and forecast the next 16 days."""
    cutoff = pd.Timestamp(cutoff)
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        batches = Parallel()(delayed(_predict_series)(key, frame, cutoff, columns, train_days)
                             for key, frame in _frames(panel, columns))
    return pd.concat([rows for rows, _ in batches], ignore_index=True), pd.DataFrame(
        fit for _, fit in batches
    )
