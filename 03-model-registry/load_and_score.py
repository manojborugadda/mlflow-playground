import argparse
from pathlib import Path

import mlflow
import pandas as pd

PRJ_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"file://{PRJ_ROOT / 'mlruns'}"
MODEL_NAME = "smartphone_returns_rf"
DATA = PRJ_ROOT / "ep1" / "smartphone_returns.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default=None, help="Stage to load (e.g., Production). If omitted, uses alias 'best'.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)

    if args.stage:
        model_uri = f"models:/{MODEL_NAME}/{args.stage}"
    else:
        model_uri = f"models:/{MODEL_NAME}@best"

    print("Loading:", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    df = pd.read_csv(DATA)
    X = df.drop(columns=["returned"]).head(10)
    preds = model.predict(X)
    print("Predictions (first 10):", preds.tolist())


if __name__ == "__main__":
    main()