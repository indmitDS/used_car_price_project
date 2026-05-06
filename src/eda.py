"""EDA and visualization module."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")

def save_plot(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_eda(df: pd.DataFrame, plots_dir: str = "plots", reports_dir: str = "reports") -> None:
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    # Summary tables
    df.describe(include="all").transpose().to_csv(f"{reports_dir}/data_summary.csv")
    missing = df.isna().mean().sort_values(ascending=False).reset_index()
    missing.columns = ["column", "missing_rate"]
    missing.to_csv(f"{reports_dir}/missing_values.csv", index=False)

    if "manufacturer" in df.columns:
        brand_summary = df.groupby("manufacturer")["price"].agg(["count", "mean", "median"]).sort_values("median", ascending=False)
        brand_summary.to_csv(f"{reports_dir}/manufacturer_price_summary.csv")

    if "condition" in df.columns:
        condition_summary = df.groupby("condition")["price"].agg(["count", "mean", "median"]).sort_values("median", ascending=False)
        condition_summary.to_csv(f"{reports_dir}/condition_price_summary.csv")

    # Price distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df["price"], bins=60, kde=True)
    plt.title("Used Car Price Distribution")
    plt.xlabel("Price")
    save_plot(f"{plots_dir}/price_distribution.png")

    # Log price distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(np.log1p(df["price"]), bins=60, kde=True)
    plt.title("Log-Transformed Price Distribution")
    plt.xlabel("log(1 + price)")
    save_plot(f"{plots_dir}/log_price_distribution.png")

    # Odometer vs price
    plt.figure(figsize=(9, 5))
    sample = df.sample(min(len(df), 10000), random_state=42)
    sns.scatterplot(data=sample, x="odometer", y="price", alpha=0.35)
    plt.title("Price vs Odometer")
    save_plot(f"{plots_dir}/price_vs_odometer.png")

    # Year vs price
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df.sample(min(len(df), 30000), random_state=42), x="year", y="price")
    plt.xticks(rotation=90)
    plt.title("Price by Vehicle Year")
    save_plot(f"{plots_dir}/price_by_year.png")

    # Correlation heatmap
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Numeric Feature Correlation Heatmap")
        save_plot(f"{plots_dir}/correlation_heatmap.png")

    # Manufacturer boxplot
    if "manufacturer" in df.columns:
        top = df["manufacturer"].value_counts().nlargest(12).index
        subset = df[df["manufacturer"].isin(top)]
        plt.figure(figsize=(11, 6))
        sns.boxplot(data=subset, x="manufacturer", y="price")
        plt.xticks(rotation=45, ha="right")
        plt.title("Price by Top Manufacturers")
        save_plot(f"{plots_dir}/price_by_manufacturer.png")

    # Condition boxplot
    if "condition" in df.columns:
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=df, x="condition", y="price")
        plt.xticks(rotation=45, ha="right")
        plt.title("Price by Condition")
        save_plot(f"{plots_dir}/price_by_condition.png")

    # Fuel type boxplot
    if "fuel" in df.columns:
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=df, x="fuel", y="price")
        plt.xticks(rotation=45, ha="right")
        plt.title("Price by Fuel Type")
        save_plot(f"{plots_dir}/price_by_fuel.png")
