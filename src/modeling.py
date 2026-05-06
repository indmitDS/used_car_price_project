"""Regression and classification modeling."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def build_regression_models(preprocessor, random_state=42):
    models = {
        "linear_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]),
        "ridge": Pipeline([
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0, random_state=random_state))
        ]),
        "lasso": Pipeline([ ("preprocessor", preprocessor),
              ("model", Lasso(alpha=1.0, max_iter=10000, random_state=42
            ))
        ]),
        "random_forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=50,
                max_depth=15,
                min_samples_split=10,
                random_state=random_state,
                n_jobs=-1
            ))
        ]),
        "gradient_boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.08,
                random_state=random_state
            ))
        ])
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1
            ))
        ])
    return models


def evaluate_regression_model(model, X_train, X_test, y_train, y_test, cv=3):
    start = time.time()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_test, preds),
        "training_seconds": time.time() - start
    }
    if cv:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=None)
        metrics["cv_rmse_mean"] = -cv_scores.mean()
        metrics["cv_rmse_std"] = cv_scores.std()
    return metrics, preds


def train_regression_models(X, y, preprocessor, config):
    rs = config["project"]["random_state"]
    test_size = config["modeling"]["test_size"]
    cv = config["modeling"]["cv_folds"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs)
    models = build_regression_models(preprocessor, rs)
    results, predictions, fitted_models = [], {}, {}
    for name, model in models.items():
        metrics, preds = evaluate_regression_model(model, X_train, X_test, y_train, y_test, cv=cv)
        metrics["model"] = name
        results.append(metrics)
        predictions[name] = preds
        fitted_models[name] = model
    results_df = pd.DataFrame(results).sort_values("rmse")
    best_name = results_df.iloc[0]["model"]
    return results_df, fitted_models[best_name], best_name, X_train, X_test, y_train, y_test, predictions


def tune_best_regressor(X_train, y_train, preprocessor, random_state=42, cv=3):
    if XGBOOST_AVAILABLE:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(objective="reg:squarederror", random_state=random_state, n_jobs=-1))
        ])
        params = {
            "model__n_estimators": [80, 120, 180],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.08, 0.12],
            "model__subsample": [0.7, 0.9],
            "model__colsample_bytree": [0.7, 0.9]
        }
    else:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=random_state, n_jobs=-1))
        ])
        params = {
            "model__n_estimators": [50, 80, 120],
            "model__max_depth": [10, 15, 20],
            "model__min_samples_split": [5, 10, 20]
        }
    search = RandomizedSearchCV(
        pipe, params, n_iter=8, cv=cv, scoring="neg_root_mean_squared_error",
        random_state=random_state, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    return search


def build_classification_models(preprocessor, random_state=42):
    models = {
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state))
        ]),
        "random_forest_classifier": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=80, max_depth=12, random_state=random_state, n_jobs=-1))
        ])
    }
    if XGBOOST_AVAILABLE:
        models["xgboost_classifier"] = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                random_state=random_state, n_jobs=-1
            ))
        ])
    return models


def evaluate_classifier(model, X_train, X_test, y_train, y_test, cv=3):
    start = time.time()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    out = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "training_seconds": time.time() - start
    }
    try:
        proba = model.predict_proba(X_test)[:, 1]
        out["roc_auc"] = roc_auc_score(y_test, proba)
    except Exception:
        out["roc_auc"] = np.nan
    if cv:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=None)
        out["cv_f1_mean"] = cv_scores.mean()
        out["cv_f1_std"] = cv_scores.std()
    return out


def train_classification_models(X, y, preprocessor, config):
    rs = config["project"]["random_state"]
    test_size = config["modeling"]["test_size"]
    cv = config["modeling"]["cv_folds"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=y)
    models = build_classification_models(preprocessor, rs)
    results, fitted = [], {}
    for name, model in models.items():
        metrics = evaluate_classifier(model, X_train, X_test, y_train, y_test, cv=cv)
        metrics["model"] = name
        results.append(metrics)
        fitted[name] = model
    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    best_name = results_df.iloc[0]["model"]
    return results_df, fitted[best_name], best_name

def evaluate_classification(model, X_test, y_test):
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import os

    y_pred = model.predict(X_test)

    # Save classification report
    report = classification_report(y_test, y_pred)
    os.makedirs("reports", exist_ok=True)

    with open("reports/classification_report.txt", "w") as f:
        f.write(report)

    # Save confusion matrix
    os.makedirs("plots", exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred,  display_labels=["Not Expensive", "Expensive"])
    plt.title("Classification Confusion Matrix")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    print("\nClassification Report:\n", report)

    return y_pred



def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
