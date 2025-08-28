from typing import Optional, Dict, Any, Tuple, Iterable, List
import time
import numpy as np
import pandas as pd
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

# ===== Supported metrics (sklearn scorers) =====
SCORING_MAP: Dict[str, str] = {
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "accuracy": "accuracy",
    # (optional extras you can enable later)
    # "balanced_accuracy": "balanced_accuracy",
    # "average_precision": "average_precision",
    # "neg_log_loss": "neg_log_loss",
}


def _ensure_metrics(metrics: Optional[Iterable[str]]) -> List[str]:
    """Return a clean list of metric names, defaulting to all supported."""
    if metrics is None:
        return list(SCORING_MAP.keys())
    m = list(dict.fromkeys(metrics))  # de-dup, preserve order
    unknown = [x for x in m if x not in SCORING_MAP]
    if unknown:
        raise ValueError(
            f"Unsupported metrics {unknown}. Choose from {list(SCORING_MAP)}."
        )
    return m


def _val_metric(est, X, y, metric: str) -> float:
    """Compute a validation metric for a fitted estimator."""
    if metric == "roc_auc":
        if hasattr(est, "predict_proba"):
            yv = est.predict_proba(X)[:, 1]
            return float(roc_auc_score(y, yv))
        elif hasattr(est, "decision_function"):
            yv = est.decision_function(X)
            return float(roc_auc_score(y, yv))
        else:
            # fallback (rare): use accuracy if no scores available
            return float(accuracy_score(y, est.predict(X)))
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


def grid_search_with_progress(
    estimator,
    param_grid: Dict[str, Any],
    X_train,
    y_train,
    *,
    scoring: str = "roc_auc",  # metric used to pick best params
    metrics: Optional[
        Iterable[str]
    ] = None,  # metrics to report (mean/std); default = all above
    cv: int = 5,
    random_state: int = 42,
    n_jobs: Optional[int] = None,
    X_val=None,
    y_val=None,
    refit: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Any, Dict[str, Any], float, Optional[Dict[str, float]]]:
    """
    Grid search with tqdm progress and a tidy multi-metric CV summary.

    Returns
    -------
    results_df : pd.DataFrame
        Sorted by mean_cv_<scoring> desc. Includes mean/std columns for each metric.
    best_estimator : fitted estimator or None
    best_params : dict
    best_cv_score : float
    val_scores : dict(metric -> score) or None
    """
    if scoring not in SCORING_MAP:
        raise ValueError(
            f"Unsupported scoring '{scoring}'. Choose from {list(SCORING_MAP)}."
        )
    metrics_list = _ensure_metrics(metrics)
    scoring_dict = {m: SCORING_MAP[m] for m in metrics_list}

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    rows = []
    grid = list(ParameterGrid(param_grid))

    pbar = tqdm(grid, desc=f"GridSearch (refit={scoring})", disable=not verbose)
    for params in pbar:
        est = clone(estimator).set_params(**params)
        t0 = time.time()
        try:
            cv_out = cross_validate(
                est,
                X_train,
                y_train,
                scoring=scoring_dict,
                cv=cv_splitter,
                n_jobs=n_jobs,
                error_score="raise",
                return_train_score=False,
            )
            row = dict(params)
            for m in metrics_list:
                m_scores = cv_out[f"test_{m}"]
                row[f"mean_cv_{m}"] = float(np.mean(m_scores))
                row[f"std_cv_{m}"] = float(np.std(m_scores))
            row["status"] = "ok"
        except Exception as e:
            row = dict(params)
            for m in metrics_list:
                row[f"mean_cv_{m}"] = np.nan
                row[f"std_cv_{m}"] = np.nan
            row["status"] = f"error: {e.__class__.__name__}"
        row["fit_seconds"] = round(time.time() - t0, 3)

        refit_key = f"mean_cv_{scoring}"
        refit_val = row.get(refit_key, np.nan)
        pbar.set_postfix(
            {refit_key: f"{refit_val:.4f}" if np.isfinite(refit_val) else "nan"}
        )
        rows.append(row)

    results_df = (
        pd.DataFrame(rows)
        .sort_values(f"mean_cv_{scoring}", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    best_params: Dict[str, Any] = {}
    best_cv_score: float = float("nan")
    best_estimator = None
    val_scores: Optional[Dict[str, float]] = None

    if len(results_df) and np.isfinite(results_df.loc[0, f"mean_cv_{scoring}"]):
        grid_keys = list(param_grid.keys())
        best_params = {k: results_df.loc[0, k] for k in grid_keys}
        best_cv_score = float(results_df.loc[0, f"mean_cv_{scoring}"])

        if refit:
            best_estimator = clone(estimator).set_params(**best_params)
            best_estimator.fit(X_train, y_train)

            if X_val is not None and y_val is not None:
                val_scores = {
                    m: _val_metric(best_estimator, X_val, y_val, m)
                    for m in metrics_list
                }

    if len(results_df):
        results_df[f"rank_{scoring}"] = (
            results_df[f"mean_cv_{scoring}"].rank(ascending=False, method="min")
        ).astype("Int64")

    return results_df, best_estimator, best_params, best_cv_score, val_scores
