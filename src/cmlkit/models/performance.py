from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

__all__ = ["plot_roc_pr", "classification_metrics_table"]


def _get_scores(model, X) -> np.ndarray:
    """
    Return continuous scores for binary classification:
      - predict_proba(X)[:, 1] if available
      - else decision_function(X)

    Needed for ROC/PR curves and AUROC/AP metrics.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise ValueError(
        "Model must support predict_proba or decision_function for ROC/PR plots."
    )


def classification_metrics_table(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    digits: int = 3,
) -> pd.DataFrame:
    """
    Build a tidy metrics table (Train vs Validation) with:
      - AUROC, F1, Precision, Recall, Accuracy

    Parameters
    ----------
    model : fitted binary classifier
    X_train, y_train, X_val, y_val : array-like
    digits : rounding for display

    Returns
    -------
    pd.DataFrame with index ["AUROC","F1","Precision","Recall","Accuracy"]
    and columns ["Train","Validation"].
    """
    # Continuous scores for AUROC (and optionally PR/AP)
    s_tr = _get_scores(model, X_train)
    s_va = _get_scores(model, X_val)

    # Class predictions (threshold 0.5 for probabilities, else 0.0 for decision scores)
    if hasattr(model, "predict_proba"):
        yhat_tr = (s_tr >= 0.5).astype(int)
        yhat_va = (s_va >= 0.5).astype(int)
    else:
        yhat_tr = (s_tr >= 0.0).astype(int)
        yhat_va = (s_va >= 0.0).astype(int)

    metrics: Dict[str, Dict[str, float]] = {
        "AUROC": {
            "Train": roc_auc_score(y_train, s_tr),
            "Validation": roc_auc_score(y_val, s_va),
        },
        "F1": {
            "Train": f1_score(y_train, yhat_tr),
            "Validation": f1_score(y_val, yhat_va),
        },
        "Precision": {
            "Train": precision_score(y_train, yhat_tr, zero_division=0),
            "Validation": precision_score(y_val, yhat_va, zero_division=0),
        },
        "Recall": {
            "Train": recall_score(y_train, yhat_tr),
            "Validation": recall_score(y_val, yhat_va),
        },
        "Accuracy": {
            "Train": accuracy_score(y_train, yhat_tr),
            "Validation": accuracy_score(y_val, yhat_va),
        },
    }

    return pd.DataFrame(metrics).T[["Train", "Validation"]].round(digits)


def plot_roc_pr(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    show: bool = True,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Plot ROC and Precision–Recall curves for Train and Validation.

    - Left: ROC (AUC in legend).
    - Right: PR (Average Precision in legend) + baseline (class prevalence).

    Returns
    -------
    matplotlib.figure.Figure if return_fig=True, else None.
    """
    # Colors
    col_train = "#a1d99b"  # pale green
    col_val = "#9ecae1"  # light blue
    col_base = "#d9d9d9"  # grey baseline

    # Scores
    s_tr = _get_scores(model, X_train)
    s_va = _get_scores(model, X_val)

    # ROC
    fpr_tr, tpr_tr, _ = roc_curve(y_train, s_tr)
    fpr_va, tpr_va, _ = roc_curve(y_val, s_va)
    auc_tr = auc(fpr_tr, tpr_tr)
    auc_va = auc(fpr_va, tpr_va)

    # PR
    p_tr, r_tr, _ = precision_recall_curve(y_train, s_tr)
    p_va, r_va, _ = precision_recall_curve(y_val, s_va)
    ap_tr = average_precision_score(y_train, s_tr)
    ap_va = average_precision_score(y_val, s_va)
    base = float(np.mean(np.concatenate([np.asarray(y_train), np.asarray(y_val)])))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    # ROC
    ax[0].plot(
        fpr_tr, tpr_tr, color=col_train, lw=2.5, label=f"Train (AUC = {auc_tr:.3f})"
    )
    ax[0].plot(
        fpr_va, tpr_va, color=col_val, lw=2.5, label=f"Validation (AUC = {auc_va:.3f})"
    )
    ax[0].plot([0, 1], [0, 1], "--", color=col_base, lw=1.5)
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("ROC Curve")
    ax[0].legend(loc="lower right", frameon=False)

    # PR
    ax[1].plot(r_tr, p_tr, color=col_train, lw=2.5, label=f"Train (AP = {ap_tr:.3f})")
    ax[1].plot(
        r_va, p_va, color=col_val, lw=2.5, label=f"Validation (AP = {ap_va:.3f})"
    )
    ax[1].hlines(base, 0, 1, colors=col_base, linestyles="--", lw=1.5)
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision–Recall Curve")
    ax[1].legend(loc="upper right", frameon=False)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1.05)

    plt.tight_layout()
    if show:
        plt.show()
    return fig if return_fig else None
