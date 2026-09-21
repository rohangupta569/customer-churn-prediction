"""Plots used to explain and compare trained models."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.pipeline import Pipeline


def create_evaluation_plots(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    comparison: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save model comparison, test curves, confusion matrix, and feature effects."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    long = comparison.melt(id_vars="model", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long, x="metric", y="score", hue="model", ax=ax)
    ax.set(title="Cross-validation model comparison", ylim=(0, 1), xlabel="", ylabel="Mean score")
    ax.legend(title="Model", frameon=False)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion matrix")
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1], color="#2962ff")
    axes[1].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[1].set_title("ROC curve")
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[2], color="#e45756")
    axes[2].set_title("Precision–recall curve")
    fig.tight_layout()
    fig.savefig(output / "model_evaluation.png", dpi=160)
    plt.close(fig)

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
        label = "Coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        label = "Importance"
    else:
        return
    effects = pd.DataFrame({"feature": names, "value": values})
    effects["absolute"] = effects["value"].abs()
    effects = effects.nlargest(15, "absolute").sort_values("value")
    effects.to_csv(output.parent / "feature_effects.csv", index=False)
    colors = np.where(effects["value"] >= 0, "#e45756", "#4c78a8")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(effects["feature"].str.replace(r"^(numeric|categorical)__", "", regex=True), effects["value"], color=colors)
    ax.set(title="Features with the strongest model effect", xlabel=label, ylabel="")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output / "feature_effects.png", dpi=160)
    plt.close(fig)
