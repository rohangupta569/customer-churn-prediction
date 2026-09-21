"""Command-line entry point."""

import argparse
import json
from pathlib import Path

from .data import generate_demo_data, load_data
from .eda import create_eda
from .modeling import predict, train


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="churn", description="Customer churn ML pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("generate", help="generate a demo dataset")
    demo.add_argument("--output", default="data/customer_churn.csv")
    demo.add_argument("--rows", type=int, default=3000)
    training = commands.add_parser("train", help="run EDA, model comparison, tuning, and evaluation")
    training.add_argument("--data", required=True)
    training.add_argument("--output-dir", default="artifacts")
    training.add_argument("--cv-folds", type=int, default=5)
    scoring = commands.add_parser("predict", help="score customers with a trained model")
    scoring.add_argument("--data", required=True)
    scoring.add_argument("--model", default="artifacts/churn_model.joblib")
    scoring.add_argument("--output", default="artifacts/predictions.csv")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "generate":
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        generate_demo_data(args.rows).to_csv(path, index=False)
        print(f"Generated {args.rows} rows at {path}")
    elif args.command == "train":
        df = load_data(args.data)
        create_eda(df, Path(args.output_dir) / "plots")
        print(json.dumps(train(df, args.output_dir, cv_folds=args.cv_folds), indent=2))
    else:
        df = load_data(args.data, require_target=False)
        result = predict(df, args.model)
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
        print(f"Saved {len(result)} predictions to {path}")


if __name__ == "__main__":
    main()
