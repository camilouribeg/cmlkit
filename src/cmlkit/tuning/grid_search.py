# --- cmlkit.tuning.grid_search (robust, XGB-friendly) ---

from typing import Optional, Dict, Any, Tuple, Iterable, List
import time, numpy as np, pandas as pd
from tqdm.auto import tqdm
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
)

# Supported metrics
SCORING_MAP: Dict[str, str] = {
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "accuracy": "accuracy",
}


def _ensure_metrics(metrics: Optional[Iterable[str]]) -> List[str]:
    if metrics is None:
        return list(SCORING_MAP.keys())
    m = list(dict.fromkeys(metrics))
    unknown = [x for x in m if x not in SCORING_MAP]
    if unknown:
        raise ValueError(
            f"Unsupported metrics {unknown}. Choose from {list(SCORING_MAP)}."
        )
    return m


def _val_metric(est, X, y, metric: str) -> float:
    # Safe evaluation (handles lack of proba / single-class folds)
    try:
        if metric == "roc_auc":
            if len(np.unique(y)) < 2:
                return np.nan
            if hasattr(est, "predict_proba"):
                s = est.predict_proba(X)[:, 1]
            elif hasattr(est, "decision_function"):
                s = est.decision_function(X)
            else:
                return accuracy_score(y, est.predict(X))
            return float(roc_auc_score(y, s))
        elif metric == "accuracy":
            return float(accuracy_score(y, est.predict(X)))
        elif metric == "recall":
            return float(recall_score(y, est.predict(X)))
        elif metric == "precision":
            return float(precision_score(y, est.predict(X)))
        elif metric == "f1":
            return float(f1_score(y, est.predict(X)))
        else:
            raise ValueError(f"Unsupported metric '{metric}'")
    except Exception:
        return np.nan


def grid_search_with_progress(
    estimator,
    param_grid: Dict[str, Any],
    X_train,
    y_train,
    *,
    scoring: str = "roc_auc",
    metrics: Optional[Iterable[str]] = None,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: Optional[int] = None,  # tip: use 1 for xgboost to avoid thread pile-up
    X_val=None,
    y_val=None,
    refit: bool = True,
    verbose: bool = True,
    backend: str = "auto",  # "auto" | "sklearn" | "manual"
) -> Tuple[pd.DataFrame, Any, Dict[str, Any], float, Optional[Dict[str, float]]]:
    """
    Grid search with tqdm and tidy multi-metric summary.
    - Uses sklearn.cross_validate when possible
    - Falls back to a manual StratifiedKFold loop if needed (e.g., XGB quirks)
    - Logs 'status' and 'error_msg' per row, plus fit time

    Returns
    -------
    results_df, best_estimator, best_params, best_cv_score, val_scores
    """
    if scoring not in SCORING_MAP:
        raise ValueError(
            f"Unsupported scoring '{scoring}'. Choose from {list(SCORING_MAP)}."
        )
    metrics_list = _ensure_metrics(metrics)
    scoring_dict = {m: SCORING_MAP[m] for m in metrics_list}

    rows: List[Dict[str, Any]] = []
    grid = list(ParameterGrid(param_grid))

    def _run_manual_cv(est_proto, params) -> Dict[str, Any]:
        cv_splitter = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )
        fold_scores = {m: [] for m in metrics_list}
        for tr_idx, va_idx in cv_splitter.split(X_train, y_train):
            est = clone(est_proto).set_params(**params)
            est.fit(X_train[tr_idx], y_train[tr_idx])
            Xv, yv = X_train[va_idx], y_train[va_idx]
            for m in metrics_list:
                fold_scores[m].append(_val_metric(est, Xv, yv, m))
        row = dict(params)
        for m in metrics_list:
            arr = np.array(fold_scores[m], dtype=float)
            row[f"mean_cv_{m}"] = float(np.nanmean(arr))
            row[f"std_cv_{m}"] = float(np.nanstd(arr))
        row["status"] = "ok"
        row["error_msg"] = ""
        return row

    use_manual = backend == "manual"
    use_sklearn = backend == "sklearn"

    pbar = tqdm(
        grid, desc=f"GridSearch (opt={scoring}, backend={backend})", disable=not verbose
    )
    for params in pbar:
        t0 = time.time()
        try:
            if use_manual:
                row = _run_manual_cv(estimator, params)
            else:
                # Try sklearn CV first
                cv_out = cross_validate(
                    clone(estimator).set_params(**params),
                    X_train,
                    y_train,
                    scoring=scoring_dict,
                    cv=StratifiedKFold(
                        n_splits=cv, shuffle=True, random_state=random_state
                    ),
                    n_jobs=n_jobs,
                    error_score="raise",
                    return_train_score=False,
                )
                row = dict(params)
                for m in metrics_list:
                    s = cv_out[f"test_{m}"]
                    row[f"mean_cv_{m}"] = float(np.mean(s))
                    row[f"std_cv_{m}"] = float(np.std(s))
                row["status"] = "ok"
                row["error_msg"] = ""
        except Exception as e:
            if backend == "auto" and not use_manual:
                # Fallback to manual loop
                try:
                    row = _run_manual_cv(estimator, params)
                except Exception as e2:
                    row = dict(params)
                    for m in metrics_list:
                        row[f"mean_cv_{m}"] = np.nan
                        row[f"std_cv_{m}"] = np.nan
                    row["status"] = "error"
                    row["error_msg"] = f"{type(e2).__name__}: {e2}"
            else:
                row = dict(params)
                for m in metrics_list:
                    row[f"mean_cv_{m}"] = np.nan
                    row[f"std_cv_{m}"] = np.nan
                row["status"] = "error"
                row["error_msg"] = f"{type(e).__name__}: {e}"

        refit_key = f"mean_cv_{scoring}"
        v = row.get(refit_key, np.nan)
        pbar.set_postfix({refit_key: f"{v:.4f}" if np.isfinite(v) else "nan"})
        row["fit_seconds"] = round(time.time() - t0, 3)
        rows.append(row)

    results_df = (
        pd.DataFrame(rows)
        .sort_values(f"mean_cv_{scoring}", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    if len(results_df):
        results_df[f"rank_{scoring}"] = (
            results_df[f"mean_cv_{scoring}"].rank(ascending=False, method="min")
        ).astype("Int64")

    best_params: Dict[str, Any] = {}
    best_cv_score: float = float("nan")
    best_estimator = None
    val_scores: Optional[Dict[str, float]] = None

    top_ok = results_df.query("status == 'ok'")
    if len(top_ok) and np.isfinite(top_ok.iloc[0][f"mean_cv_{scoring}"]):
        grid_keys = list(param_grid.keys())
        for k in grid_keys:
            best_params[k] = top_ok.iloc[0][k]
        best_cv_score = float(top_ok.iloc[0][f"mean_cv_{scoring}"])

        if refit:
            best_estimator = clone(estimator).set_params(**best_params)
            best_estimator.fit(X_train, y_train)
            if X_val is not None and y_val is not None:
                val_scores = {
                    m: _val_metric(best_estimator, X_val, y_val, m)
                    for m in metrics_list
                }

    return results_df, best_estimator, best_params, best_cv_score, val_scores
