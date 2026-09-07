import argparse
from pathlib import Path

import mlflow
from mlflow import MlflowClient

PRJ_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"file://{PRJ_ROOT / 'mlruns'}"
EXPERIMENT_NAME = "ep2_autolog_compare"
MODEL_NAME = "smartphone_returns_rf"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"Experiment {EXPERIMENT_NAME} not found. Run EP2 first.")

    # Search runs by F1 (manual metric we logged)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.f1_manual DESC", "metrics.accuracy_manual DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("No runs found in EP2 experiment.")

    best = runs[0]
    run_id = best.info.run_id
    print("Best run:", run_id, best.data.metrics)

    # Find the autologged model artifact path
    # Sklearn autolog stores under 'model' by default
    # We'll register that model
    mv = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=MODEL_NAME,
    )

    print("Registered model version:", mv.version)

    # Optionally set alias "best"
    client.set_registered_model_alias(MODEL_NAME, alias="best", version=mv.version)
    print(f"Set alias 'best' -> version {mv.version}")


if __name__ == "__main__":
    main()