import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

PRJ_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"file://{PRJ_ROOT / 'mlruns'}"
MODEL_NAME = "smartphone_returns_rf"
DATA = PRJ_ROOT / "ep1" / "smartphone_returns.csv"


def evaluate(model, X, y):
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)[:, 1]
    except Exception:
        y_prob = None
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
    }
    if y_prob is not None:
        metrics["auc"] = float(roc_auc_score(y, y_prob))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-alias", default="best")
    parser.add_argument("--baseline-alias", default="production")
    parser.add_argument("--min-f1", type=float, default=0.10)
    parser.add_argument("--min-auc", type=float, default=0.70)
    parser.add_argument("--promote-if-better", action="store_true", help="Promote candidate if strictly better than baseline by F1.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)

    df = pd.read_csv(DATA)
    X = df.drop(columns=["returned"])  # use full dataset for demo
    y = df["returned"].astype(int)

    # Load candidate and baseline
    cand_uri = f"models:/{MODEL_NAME}@{args.candidate_alias}"
    base_uri = f"models:/{MODEL_NAME}@{args.baseline_alias}"

    candidate = mlflow.sklearn.load_model(cand_uri)
    candidate_metrics = evaluate(candidate, X, y)
    print("Candidate:", candidate_metrics)

    baseline_metrics = None
    try:
        baseline = mlflow.sklearn.load_model(base_uri)
        baseline_metrics = evaluate(baseline, X, y)
        print("Baseline:", baseline_metrics)
    except Exception:
        print("No baseline available; treating candidate as first production.")

    # Gates
    passes = True
    if candidate_metrics.get("f1", 0.0) < args.min_f1:
        passes = False
    if "auc" in candidate_metrics and candidate_metrics["auc"] < args.min_auc:
        passes = False

    if baseline_metrics and args.promote_if_better:
        if candidate_metrics.get("f1", 0.0) <= baseline_metrics.get("f1", 0.0):
            passes = False

    if not passes:
        print("Gate failed. Not promoting.")
        return

    # Promote by switching alias 'production' to candidate version
    client = mlflow.MlflowClient(tracking_uri=TRACKING_URI)
    v = client.get_model_version_by_alias(MODEL_NAME, alias=args.candidate_alias)
    client.set_registered_model_alias(MODEL_NAME, alias="production", version=v.version)
    print(f"Promoted: alias 'production' -> version {v.version}")


if __name__ == "__main__":
    main()