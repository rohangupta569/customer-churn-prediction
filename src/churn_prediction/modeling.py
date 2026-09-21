"""Training, comparison, tuning, evaluation, and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import split_features_target
from .reporting import create_evaluation_plots

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def candidate_models(X: pd.DataFrame, random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([("preprocessor", build_preprocessor(X)), ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=random_state))]),
        "random_forest": Pipeline([("preprocessor", build_preprocessor(X)), ("model", RandomForestClassifier(n_estimators=250, class_weight="balanced", n_jobs=1, random_state=random_state))]),
    }


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    predicted = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y, probability)), 4),
        "confusion_matrix": confusion_matrix(y, predicted).tolist(),
    }


def train(df: pd.DataFrame, output_dir: str | Path, random_state: int = 42, cv_folds: int = 5) -> dict:
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    models = candidate_models(X_train, random_state)
    comparison = []
    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
        comparison.append({"model": name, **{metric: round(float(scores[f"test_{metric}"].mean()), 4) for metric in SCORING}})
    comparison.sort(key=lambda row: row["roc_auc"], reverse=True)
    winner = comparison[0]["model"]

    grids = {
        "logistic_regression": {"model__C": [0.1, 1.0, 10.0]},
        "random_forest": {"model__n_estimators": [200, 400], "model__max_depth": [None, 8, 14], "model__min_samples_leaf": [1, 3]},
    }
    search = GridSearchCV(models[winner], grids[winner], scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
    search.fit(X_train, y_train)
    metrics = evaluate(search.best_estimator_, X_test, y_test)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, output / "churn_model.joblib")
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(output / "model_comparison.csv", index=False)
    create_evaluation_plots(search.best_estimator_, X_test, y_test, comparison_df, output / "plots")
    report = {
        "rows": len(df), "features": X.columns.tolist(), "positive_rate": round(float(y.mean()), 4),
        "selected_model": winner, "best_parameters": search.best_params_, "cv_best_roc_auc": round(float(search.best_score_), 4),
        "test_metrics": metrics,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def predict(df: pd.DataFrame, model_path: str | Path) -> pd.DataFrame:
    model = joblib.load(model_path)
    result = df.copy()
    features = df.drop(columns=["churn", "customer_id"], errors="ignore")
    result["churn_probability"] = model.predict_proba(features)[:, 1].round(4)
    result["predicted_churn"] = np.where(model.predict(features) == 1, "yes", "no")
    return result
