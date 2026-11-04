"""Reproducible, leakage-aware Bank Marketing model comparison.

Run from the Bank-Marketing-ML-Project directory:
    python analysis/reproduce_analysis.py

The script deliberately excludes `duration`: it is only known after a call
and is unavailable for a pre-contact targeting decision. All learned
preprocessing is fitted inside each training fold through sklearn Pipelines.
"""
from pathlib import Path
import json
import warnings
import shutil

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (StratifiedKFold, cross_validate, cross_val_predict,
    train_test_split, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bank-full.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42


def make_preprocessor(numeric_columns, categorical_columns, scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_columns),
        ("categorical", categorical_pipe, categorical_columns),
    ])


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    required = {"y", "duration", "pdays"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    # Preserve the UCI observations. Do not delete zero-duration rows or
    # collapse pdays=-1 (never previously contacted) into pdays=0.
    df["previously_contacted"] = (df["pdays"] != -1).astype(int)
    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y", "duration"])
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Models are specified before examining the held-out test set.
    model_specs = {
        "Majority baseline": (DummyClassifier(strategy="most_frequent"), False),
        "Logistic Regression": (LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ), True),
        "Decision Tree": (DecisionTreeClassifier(
            class_weight="balanced", min_samples_leaf=10,
            random_state=RANDOM_STATE
        ), False),
        "Random Forest": (RandomForestClassifier(
            n_estimators=40, max_depth=12, min_samples_leaf=10, max_samples=0.6, class_weight="balanced_subsample",
            n_jobs=1, random_state=RANDOM_STATE
        ), False),
        "XGBoost": (XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0, eval_metric="logloss", n_jobs=1,
            random_state=RANDOM_STATE
        ), False),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rows, prediction_data, validation_prediction_data = [], {}, {}
    for name, (estimator, scale) in model_specs.items():
        pipe = Pipeline([
            ("preprocess", make_preprocessor(numeric, categorical, scale)),
            ("model", estimator),
        ])
        # Cross-validation uses training data only. AP is useful for the
        # imbalanced positive class; ROC-AUC is included as a secondary score.
        if name == "Majority baseline":
            cv_summary = {"cv_ap_mean": float(np.mean(cross_validate(
                pipe, X_train, y_train, cv=cv, scoring="average_precision",
                n_jobs=1, return_train_score=False)["test_score"]))}
        else:
            scores = cross_validate(
                pipe, X_train, y_train, cv=cv,
                scoring={"ap": "average_precision", "roc_auc": "roc_auc"},
                n_jobs=1, return_train_score=False,
            )
            cv_summary = {
                "cv_ap_mean": float(np.mean(scores["test_ap"])),
                "cv_ap_std": float(np.std(scores["test_ap"], ddof=1)),
                "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            }
        # Out-of-fold probabilities are generated only from training data.
        # These are used by the website threshold explorer; the test set remains
        # reserved for final evaluation.
        if name != "Majority baseline":
            oof_prob = cross_val_predict(
                pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
            )[:, 1]
        else:
            oof_prob = np.zeros(len(y_train), dtype=float)
        validation_prediction_data[name] = {
            "y_true": y_train.astype(int).tolist(),
            "probability": np.asarray(oof_prob, dtype=float).tolist(),
        }

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba(X_test)[:, 1]
        else:
            prob = pred.astype(float)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        row = {
            "model": name,
            **cv_summary,
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "test_precision": float(precision_score(y_test, pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, prob)),
            "test_average_precision": float(average_precision_score(y_test, prob)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
        rows.append(row)
        prediction_data[name] = {
            "y_true": y_test.astype(int).tolist(),
            "probability": np.asarray(prob, dtype=float).tolist(),
            "prediction_at_0_5": np.asarray(pred, dtype=int).tolist(),
        }
        print(f"{name:22s} accuracy={row['test_accuracy']:.4f} "
              f"F1={row['test_f1']:.4f} AP={row['test_average_precision']:.4f} "
              f"ROC-AUC={row['test_roc_auc']:.4f}")

    # A modest, training-only grid search for Random Forest. The test set is
    # not involved in selecting hyperparameters.
    rf_base = Pipeline([
        ("preprocess", make_preprocessor(numeric, categorical, False)),
        ("model", RandomForestClassifier(
            class_weight="balanced_subsample", max_samples=0.6,
            n_jobs=1, random_state=RANDOM_STATE
        )),
    ])
    rf_search = GridSearchCV(
        rf_base,
        param_grid={
            "model__n_estimators": [40, 80],
            "model__max_depth": [10, 14],
            "model__min_samples_leaf": [5, 10],
        },
        scoring="average_precision",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    rf_search.fit(X_train, y_train)
    tuned_rf = rf_search.best_estimator_
    pd.DataFrame(rf_search.cv_results_).to_csv(OUT / "random_forest_grid_search.csv", index=False)
    tuned_pred = tuned_rf.predict(X_test)
    tuned_prob = tuned_rf.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, tuned_pred, labels=[0, 1]).ravel()
    rows.append({
        "model": "Tuned Random Forest",
        "cv_ap_mean": float(rf_search.best_score_),
        "cv_ap_std": float(rf_search.cv_results_["std_test_score"][rf_search.best_index_]),
        "cv_roc_auc_mean": np.nan,
        "test_accuracy": float(accuracy_score(y_test, tuned_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, tuned_pred)),
        "test_precision": float(precision_score(y_test, tuned_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, tuned_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, tuned_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, tuned_prob)),
        "test_average_precision": float(average_precision_score(y_test, tuned_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    })
    prediction_data["Tuned Random Forest"] = {
        "y_true": y_test.astype(int).tolist(),
        "probability": tuned_prob.astype(float).tolist(),
        "prediction_at_0_5": tuned_pred.astype(int).tolist(),
    }
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "model_comparison.csv", index=False)
    with open(OUT / "test_predictions.json", "w") as f:
        json.dump(prediction_data, f)
    with open(OUT / "validation_oof_predictions.json", "w") as f:
        json.dump(validation_prediction_data, f)

    # Percentile bootstrap intervals quantify sampling uncertainty on the
    # fixed held-out test set. They do not account for split/model-selection
    # uncertainty and are not used to tune models or thresholds.
    rng = np.random.default_rng(RANDOM_STATE)
    ci_rows = []
    y_test_array = y_test.to_numpy()
    for name, item in prediction_data.items():
        probs = np.asarray(item["probability"], dtype=float)
        preds = (probs >= 0.5).astype(int)
        boot = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
        for _ in range(1000):
            idx = rng.integers(0, len(y_test_array), len(y_test_array))
            yb, pb, prb = y_test_array[idx], probs[idx], preds[idx]
            if len(np.unique(yb)) < 2:
                continue
            boot["accuracy"].append(accuracy_score(yb, prb))
            boot["precision"].append(precision_score(yb, prb, zero_division=0))
            boot["recall"].append(recall_score(yb, prb, zero_division=0))
            boot["f1"].append(f1_score(yb, prb, zero_division=0))
            boot["roc_auc"].append(roc_auc_score(yb, pb))
        for metric, values in boot.items():
            ci_rows.append({
                "model": name, "metric": metric,
                "lower_95": float(np.percentile(values, 2.5)),
                "upper_95": float(np.percentile(values, 97.5)),
                "bootstrap_replicates": len(values),
            })
    pd.DataFrame(ci_rows).to_csv(OUT / "test_metric_bootstrap_ci.csv", index=False)

    # Descriptive error analysis for Random Forest at threshold 0.50.
    # Group rates are descriptive and should not be interpreted causally.
    rf_item = prediction_data["Random Forest"]
    rf_pred = (np.asarray(rf_item["probability"]) >= 0.5).astype(int)
    error_frame = X_test.reset_index(drop=True).copy()
    error_frame["actual"] = y_test.reset_index(drop=True).to_numpy()
    error_frame["predicted"] = rf_pred
    error_frame["error_type"] = np.select(
        [(error_frame.actual == 0) & (error_frame.predicted == 1),
         (error_frame.actual == 1) & (error_frame.predicted == 0)],
        ["false_positive", "false_negative"], default="correct"
    )
    error_frame["age_band"] = pd.cut(
        error_frame["age"], bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "66+"],
        include_lowest=True
    )
    group_rows = []
    for feature in ["job", "marital", "education", "contact", "poutcome", "age_band"]:
        for group, part in error_frame.groupby(feature, dropna=False, observed=False):
            n = len(part)
            positives = int((part.actual == 1).sum())
            group_rows.append({
                "feature": feature, "group": str(group), "n": n,
                "actual_subscribers": positives,
                "false_positive_count": int((part.error_type == "false_positive").sum()),
                "false_negative_count": int((part.error_type == "false_negative").sum()),
                "false_positive_rate_among_actual_no": (
                    float(((part.actual == 0) & (part.predicted == 1)).sum() / max((part.actual == 0).sum(), 1))
                ),
                "false_negative_rate_among_actual_yes": (
                    float(((part.actual == 1) & (part.predicted == 0)).sum() / max(positives, 1))
                ),
            })
    pd.DataFrame(group_rows).to_csv(OUT / "random_forest_error_by_group.csv", index=False)
    # Keep both the editable website source and the GitHub Pages copy current.
    for site_name in ("website", "docs"):
        site_data = ROOT / site_name / "data"
        site_data.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT / "test_predictions.json", site_data / "test_predictions.json")
        shutil.copy2(OUT / "validation_oof_predictions.json", site_data / "validation_oof_predictions.json")
        shutil.copy2(OUT / "test_metric_bootstrap_ci.csv", site_data / "test_metric_bootstrap_ci.csv")
        shutil.copy2(OUT / "random_forest_error_by_group.csv", site_data / "random_forest_error_by_group.csv")
        shutil.copy2(OUT / "random_forest_grid_search.csv", site_data / "random_forest_grid_search.csv")
    metadata = {
        "dataset": "UCI Bank Marketing bank-full.csv",
        "observations": int(len(df)),
        "positive_count": int(y.sum()),
        "negative_count": int((1-y).sum()),
        "positive_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "split": "stratified random 80/20 split",
        "cv": "3-fold stratified CV on training partition only",
        "target": "y (yes=1, no=0)",
        "excluded_feature": "duration (not available before the call)",
        "pdays_handling": "-1 retained; previously_contacted indicator added",
        "preprocessing": "imputation and one-hot encoding fitted inside each pipeline/fold",
        "resampling": "none; class weighting used for classifiers where supported",
        "note": "Test set used only for final reporting; not used for model selection or threshold selection.",
        "threshold_explorer": "Uses out-of-fold training predictions, not held-out test predictions.",
        "hyperparameter_tuning": "Random Forest grid search on training data only using 3-fold stratified CV and average precision.",
        "random_forest_best_params": rf_search.best_params_,
        "uncertainty": "Percentile bootstrap 95% intervals on held-out test metrics; conditional on this split and fitted models.",
        "error_analysis": "Random Forest false-positive and false-negative rates summarized descriptively across selected customer groups.",
    }
    with open(OUT / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved outputs to: {OUT}")


if __name__ == "__main__":
    main()
