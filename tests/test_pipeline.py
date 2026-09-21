import json

from churn_prediction.data import generate_demo_data, split_features_target
from churn_prediction.modeling import predict, train


def test_demo_data_is_reproducible():
    first = generate_demo_data(120, random_state=7)
    second = generate_demo_data(120, random_state=7)
    assert first.equals(second)
    assert {"customer_id", "churn", "monthly_charges"}.issubset(first.columns)


def test_labels_are_binary():
    X, y = split_features_target(generate_demo_data(120))
    assert "customer_id" not in X
    assert set(y.unique()) == {0, 1}


def test_end_to_end_training_and_prediction(tmp_path):
    df = generate_demo_data(240)
    report = train(df, tmp_path, cv_folds=2)
    assert 0 <= report["test_metrics"]["roc_auc"] <= 1
    assert (tmp_path / "churn_model.joblib").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["selected_model"]
    scored = predict(df.head(8).drop(columns="churn"), tmp_path / "churn_model.joblib")
    assert scored["churn_probability"].between(0, 1).all()
    assert len(scored) == 8
