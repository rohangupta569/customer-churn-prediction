"""Dataset loading, validation, and reproducible demo-data generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "churn"
ID_COLUMNS = ("customer_id",)


def generate_demo_data(n_rows: int = 3_000, random_state: int = 42) -> pd.DataFrame:
    """Create a realistic, intentionally imperfect telecom churn dataset."""
    if n_rows < 100:
        raise ValueError("n_rows must be at least 100")
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(0, 73, n_rows)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n_rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["fiber", "dsl", "none"], n_rows, p=[0.47, 0.43, 0.10])
    payment = rng.choice(["electronic-check", "bank-transfer", "credit-card", "mailed-check"], n_rows)
    support_calls = np.clip(rng.poisson(1.6, n_rows), 0, 8)
    monthly = np.round(np.clip(rng.normal(68, 27, n_rows), 18, 125), 2)
    usage = np.round(np.clip(rng.normal(125, 55, n_rows), 0, 350), 1)
    late_payments = np.clip(rng.poisson(0.8, n_rows), 0, 6)
    senior = rng.choice(["yes", "no"], n_rows, p=[0.17, 0.83])
    partner = rng.choice(["yes", "no"], n_rows)
    paperless = rng.choice(["yes", "no"], n_rows, p=[0.62, 0.38])

    logit = (
        -1.45
        + 1.15 * (contract == "month-to-month")
        + 0.75 * (internet == "fiber")
        + 0.35 * (payment == "electronic-check")
        + 0.24 * support_calls
        + 0.22 * late_payments
        + 0.012 * (monthly - 65)
        - 0.033 * tenure
        + 0.30 * (senior == "yes")
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n_rows) < probability, "yes", "no")
    total = np.round(monthly * tenure + rng.normal(0, 80, n_rows), 2)
    total = np.clip(total, 0, None)

    df = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, n_rows + 1)],
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "avg_monthly_usage_gb": usage,
        "support_calls_6m": support_calls,
        "late_payments_12m": late_payments,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "senior_citizen": senior,
        "has_partner": partner,
        "paperless_billing": paperless,
        TARGET: churn,
    })
    # Add a small amount of missing data to exercise the preprocessing pipeline.
    for column in ["total_charges", "avg_monthly_usage_gb", "payment_method"]:
        df.loc[rng.random(n_rows) < 0.02, column] = np.nan
    return df


def load_data(path: str | Path, require_target: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if df.empty:
        raise ValueError("Dataset is empty")
    if require_target and TARGET not in df.columns:
        raise ValueError(f"Dataset must contain a '{TARGET}' column")
    return df.drop_duplicates().reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in df:
        raise ValueError(f"Missing target column: {TARGET}")
    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    if y.isna().any():
        bad = sorted(df.loc[y.isna(), TARGET].astype(str).unique())
        raise ValueError(f"Unsupported churn labels: {bad}")
    if y.nunique() != 2:
        raise ValueError("Training data must contain both churn classes")
    features = df.drop(columns=[TARGET, *[c for c in ID_COLUMNS if c in df.columns]])
    return features, y.astype(int)
