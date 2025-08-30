from typing import Optional, Dict, Any, Tuple, Iterable, List
import time, numpy as np, pandas as pd
from tqdm.auto import tqdm
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate


def grid_search_with_progress(
    estimator,
    param_grid: Dict[str, Any],
    X_train,
    y_train,
    *,
    scoring: str = "roc_auc",  # metric used to pick best params
    metrics: Optional[Iterable[str]] = None,  # metrics to report (mean/std)
    cv: int = 5,
    random_state: int = 42,
    n_jobs: Optional[int] = None,
    X_val=None,
    y_val=None,
    refit: bool = True,
    verbose: bool = True,
    backend: str = "auto",  # "auto" | "sklearn" | "manual"
) -> Tuple[pd.DataFrame, Any, Dict[str, Any], float, Optional[Dict[str, float]]]:
    """
    Grid search with tqdm and tidy multi-metric summary.

    Returns
    -------
    results_df : pd.DataFrame
        Sorted by mean_cv_<scoring> desc. Includes mean/std columns for each metric.
        Adds 'status', 'fit_seconds', and 'error_msg' ('' if ok).
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

    rows: List[Dict[str, Any]] = []
    grid = list(ParameterGrid(param_grid))

    def _run_manual_cv(est_proto, params):
        """Manual CV loop (no sklearn tags). Returns row dict."""
        cv_splitter = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )
        fold_scores = {m: [] for m in metrics_list}
        for tr_idx, va_idx in cv_splitter.split(X_train, y_train):
            est = clone(est_proto).set_params(**params)
            est.fit(X_train[tr_idx], y_train[tr_idx])
            for m in metrics_list:
                # AUC needs both classes; guard with NaN if needed
                if m == "roc_auc" and len(np.unique(y_train[va_idx])) < 2:
                    fold_scores[m].append(np.nan)
                else:
                    fold_scores[m].append(
                        _val_metric(est, X_train[va_idx], y_train[va_idx], m)
                    )
        row = dict(params)
        for m in metrics_list:
            arr = np.array(fold_scores[m], dtype=float)
            row[f"mean_cv_{m}"] = float(np.nanmean(arr))
            row[f"std_cv_{m}"] = float(np.nanstd(arr))
        row["status"] = "ok"
        row["error_msg"] = ""
        return row

    pbar = tqdm(
        grid, desc=f"GridSearch (opt={scoring}, backend={backend})", disable=not verbose
    )
    use_manual = backend == "manual"
    use_sklearn = backend == "sklearn"

    for params in pbar:
        t0 = time.time()
        row: Dict[str, Any]
        try:
            if use_manual:
                row = _run_manual_cv(estimator, params)
            else:
                # try sklearn cross_validate
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
                    m_scores = cv_out[f"test_{m}"]
                    row[f"mean_cv_{m}"] = float(np.mean(m_scores))
                    row[f"std_cv_{m}"] = float(np.std(m_scores))
                row["status"] = "ok"
                row["error_msg"] = ""
        except Exception as e:
            if backend == "auto" and not use_manual:
                # fallback to manual
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
        refit_val = row.get(refit_key, np.nan)
        pbar.set_postfix(
            {refit_key: f"{refit_val:.4f}" if np.isfinite(refit_val) else "nan"}
        )
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
