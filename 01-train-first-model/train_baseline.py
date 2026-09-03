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

ROOT = Path(__file__).parent
DATA = ROOT / "smartphone_returns.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# Mlflow setup 
mlflow.set_experiment("ep1_baseline")

# Load the data
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

# Model and hyperparameter
n_estimators = int(os.getenv("RF_N_ESTIMATORS", 200))
max_depth = int(os.getenv("RF_MAX_DEPTH", 12))
RANDOM_STATE = 333


model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"
)

clf = Pipeline(steps=[("prep", preprocess), ("rf", model)])

# Train Test split 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

with mlflow.start_run(run_name="baseline_rf"):
    # Log parmeters
    mlflow.log_param("n_estimator", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Fit 
    clf.fit(X_train, y_train)

    # Eval
    y_pred = clf.predict(X_test)
    y_prob = None
    try:
        y_prob = clf.predict_proba(X_test)[:,1]
    except Exception:
        pass

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("Accuracy", float(acc))
    mlflow.log_metric("F1", float(f1))
    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
        mlflow.log_metric("auc", float(auc))


        # ROC curve
        from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
        fig_roc,ax_roc = plt.subplots(figsize= (5,4))
        RocCurveDisplay.from_predictions(y_test, y_prob, ax= ax_roc)
        roc_path = ARTIFACTS / "roc_curve.png"
        fig_roc.tight_layout()
        fig_roc.savefig(roc_path, dpi= 160)
        plt.close(fig_roc)
        mlflow.log_artifact(str(roc_path))

    # save artifacts : class balance
    class_balance = y.value_counts(normalize=True).rename_axis("class").reset_index(name="pct")
    class_balance_path = ARTIFACTS / "class_balance.csv"
    class_balance.to_csv(class_balance_path, index=False)
    mlflow.log_artifact(str(class_balance_path))


    # Confusion matrix artifact
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize = (4,4))
    im = ax_cm.imshow(cm, cmap = "Blues")
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Pridected")
    ax_cm.set_ylabel("Actual")

    for (i, j), v in np.ndenumerate(cm):
        ax_cm.text(j,i, int(v), ha="center", va = "center")
    fig_cm.colorbar(im, ax= ax_cm, fraction=0.046, pad=0.04)
    cm_path = ARTIFACTS / "confusion_matrix.png"
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path, dpi=160)
    plt.close(fig_cm)
    mlflow.log_artifact(str(cm_path))

    # Log model
    in_examples = X_train.head(5)
    signature = infer_signature(in_examples, clf.predict(in_examples))
    mlflow.sklearn.log_model(clf, name= "model", signature=signature,input_example=in_examples)

    print({"accuracy": acc, "f1": f1})