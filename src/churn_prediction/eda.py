"""Lightweight exploratory analysis with saved, reproducible plots."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_eda(df: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df["churn"] = plot_df["churn"].astype(str).str.lower()
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.countplot(data=plot_df, x="churn", ax=axes[0], color="#3973ac")
    axes[0].set_title("Customer churn distribution")
    if "tenure_months" in plot_df:
        sns.histplot(data=plot_df, x="tenure_months", hue="churn", bins=24, multiple="stack", ax=axes[1])
        axes[1].set_title("Tenure by churn status")
    fig.tight_layout()
    fig.savefig(output / "churn_overview.png", dpi=160)
    plt.close(fig)

    if "contract_type" in plot_df:
        rates = plot_df.assign(churn_flag=plot_df["churn"].isin(["yes", "1", "true"]).astype(int)).groupby("contract_type", as_index=False)["churn_flag"].mean()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=rates, x="contract_type", y="churn_flag", ax=ax, color="#e57252")
        ax.set(title="Churn rate by contract", ylabel="Churn rate", xlabel="Contract type", ylim=(0, 1))
        fig.tight_layout()
        fig.savefig(output / "churn_by_contract.png", dpi=160)
        plt.close(fig)
