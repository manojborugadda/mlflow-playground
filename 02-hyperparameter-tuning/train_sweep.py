'''
In this file we demonstrate how to use MLflow to perform hyperparameter tuning using a sweep. 
We will train a Random Forest model on the Titanic dataset, and we will sweep over different hyperparameter configurations to find the best model.
------Sweeps 3 RF configs, uses MLflow autolog, nested runs--------

In 01-train-first-model we have seen we did some lots of manual logging , 
in this module we are going to run multiple RF model training each with their own config & 
with their own pipeline , and track which part of Dataset are being used for what model
We will be learning about Mlflow evaluates
'''

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlflow.models import infer_signature
import mlflow
import os
from pathlib import Path
import mlflow.sklearn
import numpy as np
import pandas as pd


# Paths

EP_ROOT = Path(__file__).parent
PRJ_ROOT = EP_ROOT.parent
DATA = PRJ_ROOT/ "01-train-first-model" / "smartphone_returns.csv"
ARTIFACTS = EP_ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

TRACKING_DB = PRJ_ROOT / "01-train-first-model" / "mlflow.db"
ARTIFACT_ROOT = PRJ_ROOT / "01-train-first-model" / "mlruns"
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
mlflow.set_experiment("ep2_autolog_compare")
df = pd.read_csv(DATA)

# Split 
X = df.drop(columns=["returned"])
y = df['returned'].astype(int)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.to_list()

# preprocessing 

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ]
)
RANDOM_STATE= 333
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Enable the auto loggin (sklearn)
mlflow.sklearn.autolog(log_input_examples=True, log_models=True, silent=True)

# Small sweep of configs

configs = [
    {"n_estimators": 150, "max_depth": 10},
    {"n_estimators": 300, "max_depth": 12},
    {"n_estimators": 450, "max_depth": None}
]
result = []

with mlflow.start_run(run_name="rf_sweep_parent") as parent_run:
    try:
        ds_train = mlflow.data.from_pandas(
            X_train.assign(returned=y_train),
            source = str(DATA),
            name = "smartphone_returned_train",

        )
        mlflow.log_input(ds_train, context="training")
    except Exception as e:
        print(f"Error {e}")
    for cfg in configs:
        with mlflow.start_run(run_name=f"rf_{cfg['n_estimators']}x{cfg['max_depth']}", nested=True):
            model = RandomForestClassifier(
                n_estimators=cfg['n_estimators'],
                max_depth=cfg['max_depth'],
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
            clf = Pipeline(steps=[("prep", preprocess), ("rf", model)])

            # Fit and evalute
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
    
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("f1", float(f1))
            mlflow.set_tags({
                "stage":"hyperparameter_tuning",
                "mode_family": "RandomForest",
                "purpose": "hyperparam_sweep"
            })

            # Automatic evaulation
            try:
                eval_df = X_test.assign(returend = y_test)
                mlflow.evaluate(
                    clf,
                    data=eval_df,
                    targets="returned",
                    model_type="classifier",
                    evaluators= ["default"]
                )
            except Exception as e:
                print(f"error : {e}")

            result.append( {"n_estimators": cfg['n_estimators'], "max_depth": cfg['max_depth'], "accuracy": float(acc), "f1": float(f1)})


