"""Data cleaning functions for used car data."""
from __future__ import annotations
import pandas as pd


def clean_vehicle_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean vehicles dataset and create vehicle_age feature."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    needed = [
        "price", "year", "manufacturer", "model", "condition", "cylinders",
        "fuel", "odometer", "title_status", "transmission", "drive", "type", "paint_color", "state"
    ]
    existing = [c for c in needed if c in df.columns]
    df = df[existing].copy()

    for col in ["price", "year", "odometer"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rules = config["cleaning"]
    df = df.dropna(subset=["price", "year", "odometer"])
    df = df[(df["price"] >= rules["min_price"]) & (df["price"] <= rules["max_price"])]
    df = df[(df["year"] >= rules["min_year"]) & (df["year"] <= rules["max_year"])]
    df = df[(df["odometer"] >= 0) & (df["odometer"] <= rules["max_odometer"])]

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown").astype(str).str.lower().str.strip()
        top_values = df[col].value_counts().nlargest(30).index
        df[col] = df[col].where(df[col].isin(top_values), "other")

    df["vehicle_age"] = config["cleaning"]["max_year"] - df["year"]
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def add_classification_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create binary target: expensive vs not expensive."""
    df = df.copy()
    target_name = config["modeling"]["classification_target_name"]
    threshold_config = config["modeling"].get("expensive_threshold", "median")
    threshold = df["price"].median() if threshold_config == "median" else float(threshold_config)
    df[target_name] = (df["price"] > threshold).astype(int)
    return df
