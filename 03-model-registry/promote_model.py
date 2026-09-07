import argparse
from pathlib import Path

from mlflow import MlflowClient

PRJ_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"file://{PRJ_ROOT / 'mlruns'}"
MODEL_NAME = "smartphone_returns_rf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["None", "Staging", "Production", "Archived"], required=True)
    parser.add_argument("--version", type=int, default=None, help="Version to promote. If omitted, uses alias 'best'.")
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=TRACKING_URI)

    if args.version is None:
        # Resolve alias 'best'
        v = client.get_model_version_by_alias(MODEL_NAME, alias="best")
        version = int(v.version)
    else:
        version = args.version

    client.transition_model_version_stage(MODEL_NAME, str(version), stage=args.stage, archive_existing_versions=False)
    print(f"Transitioned {MODEL_NAME} v{version} -> {args.stage}")

    # Maintain alias mapping per stage for convenience
    alias = args.stage.lower() if args.stage != "None" else "none"
    client.set_registered_model_alias(MODEL_NAME, alias=alias, version=version)
    print(f"Alias '{alias}' -> v{version}")


if __name__ == "__main__":
    main()