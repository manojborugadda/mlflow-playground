# Model Registry Demo

This folder demonstrates how MLflow Model Registry is used in an MLOps workflow to manage trained models as versioned, reusable assets.

## What this folder contains

- `select_and_register.py`
  Selects the best run from an experiment, registers it as a model, and assigns the `best` alias.

- `promote_model.py`
  Promotes a registered model version to a lifecycle stage such as `Staging`, `Production`, or `Archived`.

- `load_and_score.py`
  Loads the model from the registry using the model name and stage/alias, then scores sample data.

## What is happening here

This workflow follows the usual ML lifecycle:

1. Train a model and log metrics/artifacts in MLflow experiments.
2. Choose the best-performing run.
3. Register that model in MLflow Model Registry.
4. Give it a version number and a name such as `smartphone_returns_rf`.
5. Use aliases and stages to control which version is considered current or production-ready.
6. Load the approved model from the registry during inference or deployment instead of relying on local files or notebook outputs.

## Why Model Registry matters in MLOps

In real ML systems, models are not just files. They are production assets that need:

- version control
- reproducibility
- auditability
- rollout control
- rollback support

Model Registry solves this by keeping a clean, tracked catalog of model versions.

## How the lifecycle works

### 1. Experiment and training
A model is trained and evaluated. Metrics like accuracy and F1 are logged. The run metadata, parameters, and artifacts are stored in MLflow.

### 2. Registration
The best run is registered under a model name. MLflow creates a new model version, for example:

- `smartphone_returns_rf` version 1
- `smartphone_returns_rf` version 2

Each version records the exact trained model artifact and metadata.

### 3. Staging and promotion
A model can move through lifecycle stages:

- `None` — not in a formal stage
- `Staging` — testing/validation
- `Production` — approved for serving
- `Archived` — no longer active but retained for history

This allows teams to safely promote a model only after validation.

### 4. Aliases
Aliases like `best` make it easy to reference a model without remembering version numbers.

Example:

- `models:/smartphone_returns_rf@best`
- `models:/smartphone_returns_rf/Production`

This is useful for deployment and inference pipelines.

### 5. Inference
At serving time, the application loads the model from the registry using its name and stage/alias. This ensures the deployed system uses the approved model version rather than a random or outdated file.

## Simple explanation in interview language

"Model Registry is the versioned store for trained machine learning models. It helps us track which model is the best, which version is already approved for production, and which one should be used for inference. In practical MLOps terms, it is like a controlled catalog for models, similar to how a software repository manages code versions."

## In one line

This folder shows how MLflow helps move a trained model from experimentation to a governed, versioned, production-ready asset.
