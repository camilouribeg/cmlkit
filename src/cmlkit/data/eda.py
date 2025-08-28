# target_relations.py (fixed)
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype, is_bool_dtype


def plot_feature_vs_binary_target(
    df: pd.DataFrame,
    target: str,
    feature: str,
    *,
    bins: int = 20,
    top_n: int = 15,
    dropna: bool = True,
    figsize_numeric: tuple[int, int] = (12, 4),
    figsize_categorical: tuple[int, int] = (14, 5),
    palette: dict | None = None,
) -> dict:
    if target not in df.columns or feature not in df.columns:
        raise KeyError(f"Columns not found: {target!r} or {feature!r}")

    data = df[[target, feature]].copy()
    if dropna:
        data = data.dropna(subset=[target, feature])

    # Coerce/validate binary target
    t = data[target]
    if is_bool_dtype(t):
        data[target] = t.astype(int)
    elif is_numeric_dtype(t):
        uniq = pd.unique(t.dropna())
        allowed = {0, 1, 0.0, 1.0}
        if not set(map(float, uniq)).issubset(allowed):
            raise ValueError(
                f"Target {target!r} must be binary (0/1). Found values: {sorted(map(float, uniq))[:6]}"
            )
        data[target] = t.astype(int)
    else:
        tlow = t.astype(str).str.strip().str.lower()
        for mapping in [
            {"no": 0, "yes": 1},
            {"negative": 0, "positive": 1},
            {"false": 0, "true": 1},
            {"not_drafted": 0, "drafted": 1},
            {"0": 0, "1": 1},
        ]:
            if set(tlow.unique()).issubset(mapping.keys()):
                data[target] = tlow.map(mapping).astype(int)
                break
        else:
            raise ValueError(
                f"Cannot coerce non-numeric target {target!r} to binary. Convert it to 0/1 before calling."
            )

    if palette is None:
        palette = {0: "#d62728", 1: "#2ca02c"}  # red / green

    # Numeric feature
    if is_numeric_dtype(data[feature]):
        fig, axes = plt.subplots(1, 2, figsize=figsize_numeric)

        sns.boxplot(
            data=data,
            x=target,
            y=feature,
            palette=[palette.get(0, "#d62728"), palette.get(1, "#2ca02c")],
            ax=axes[0],
        )
        axes[0].set_title(f"{feature} vs {target} (Boxplot)")
        axes[0].set_xlabel(target)
        axes[0].set_ylabel(feature)
        axes[0].grid(True, axis="y", alpha=0.3)

        hue_order = (
            [0, 1]
            if set(data[target].unique()) == {0, 1}
            else sorted(data[target].unique())
        )
        sns.histplot(
            data=data,
            x=feature,
            hue=target,
            bins=bins,
            multiple="stack",
            palette=palette,
            hue_order=hue_order,
            ax=axes[1],
        )
        axes[1].set_title(f"Distribution of {feature} by {target}")
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel("Count")
        axes[1].grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        return {"kind": "numeric", "fig": fig, "axes": axes, "data_summary": None}

    # Categorical feature (distribution-only)
    cat = data[feature].astype("string").fillna("NA")
    counts_full = cat.value_counts(dropna=False)
    keep = set(counts_full.head(top_n).index)
    cat_trim = cat.where(cat.isin(keep), other="Other")

    counts = cat_trim.value_counts(dropna=False).sort_values(ascending=False)
    order = counts.index.tolist()
    total = counts.sum()
    pct = (counts / max(total, 1)) * 100.0

    summary = pd.DataFrame({feature: order, "count": counts.values, "pct": pct.values})

    fig, axes = plt.subplots(1, 2, figsize=figsize_categorical)

    # Left: percentage
    sns.barplot(
        x=feature,
        y="pct",
        data=summary,
        order=order,
        ax=axes[0],
        color="#4c78a8",
        edgecolor="black",
    )
    axes[0].set_title(f"{feature} distribution (% of total)")
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Percentage")
    axes[0].set_ylim(0, max(100, (summary["pct"].max() * 1.15)))
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", labelrotation=30)  # <-- fixed
    for lbl in axes[0].get_xticklabels():
        lbl.set_ha("right")  # <-- fixed

    # annotate %
    for p in axes[0].patches:
        h = p.get_height()
        if np.isfinite(h) and h > 0:
            axes[0].annotate(
                f"{h:.1f}%",
                (p.get_x() + p.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 2),
                textcoords="offset points",
            )

    # Right: counts
    sns.barplot(
        x=feature,
        y="count",
        data=summary,
        order=order,
        ax=axes[1],
        color="#72b7b2",
        edgecolor="black",
    )
    axes[1].set_title(f"{feature} distribution (counts)")
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Count")
    axes[1].set_ylim(0, max(summary["count"].max() * 1.15, 1))
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", labelrotation=30)  # <-- fixed
    for lbl in axes[1].get_xticklabels():
        lbl.set_ha("right")  # <-- fixed

    fig.tight_layout()
    return {"kind": "categorical", "fig": fig, "axes": axes, "data_summary": summary}
