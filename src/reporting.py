"""Reporting utilities."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.preprocessing import get_feature_names
from sklearn.inspection import permutation_importance



def plot_top_feature_importance(feature_importance_path, output_path, top_n=20):
    """
    Plots top model-based feature importances.
    Uses reports/feature_importance.csv.
    """
    fi = pd.read_csv(feature_importance_path)
    fi = fi.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    plt.barh(fi["feature"][::-1], fi["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Predictors of Used Car Price")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_permutation_importance(model, X_test, y_test, output_path, top_n=15):
    """
    More reliable model-output plot.
    Shows how much model performance drops when each predictor is shuffled.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": result.importances_mean
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.xlabel("Increase in RMSE When Feature Is Shuffled")
    plt.title(f"Permutation Importance: Top {top_n} Price Drivers")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return importance_df


def save_results_table(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_actual_vs_predicted(y_true, y_pred, path: str) -> None:
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.35)
    low = min(np.min(y_true), np.min(y_pred))
    high = max(np.max(y_true), np.max(y_pred))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_residuals(y_true, y_pred, path: str) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_feature_importance(model, numeric_features, categorical_features, path: str, top_n: int = 30) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(model.named_steps["model"], "feature_importances_"):
        return
    preprocessor = model.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)
    importances = model.named_steps["model"].feature_importances_
    n = min(len(feature_names), len(importances))
    df = pd.DataFrame({"feature": feature_names[:n], "importance": importances[:n]})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(path, index=False)

    plt.figure(figsize=(9, 7))
    sns.barplot(data=df.head(top_n), x="importance", y="feature")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(path.replace(".csv", ".png"), dpi=150)
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, path: str) -> None:
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title("Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_executive_summary(regression_results, classification_results, path: str) -> None:
    best_reg = regression_results.sort_values("rmse").iloc[0]
    best_cls = classification_results.sort_values("f1", ascending=False).iloc[0]
    text = f"""# Executive Summary: What Drives the Price of a Car?

## Best Regression Model
- Model: {best_reg['model']}
- RMSE: {best_reg['rmse']:.2f}
- MAE: {best_reg['mae']:.2f}
- R²: {best_reg['r2']:.4f}

## Best Classification Model
The classification task predicts whether a vehicle is above the median price.
- Model: {best_cls['model']}
- Accuracy: {best_cls['accuracy']:.4f}
- F1 Score: {best_cls['f1']:.4f}
- ROC-AUC: {best_cls.get('roc_auc', np.nan):.4f}

## Business Answer
The evidence from exploratory analysis and machine learning suggests that car price is primarily driven by vehicle age/year, odometer mileage, manufacturer/brand, condition, fuel type, transmission, and body type. Newer cars with lower mileage generally sell for more. Brand and condition create additional price premiums. Tree-based models capture nonlinear interactions among these factors better than purely linear models.
"""
    Path(path).write_text(text, encoding="utf-8")
