"""Preprocessing utilities."""
from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def split_features(df, target="price", drop_cols=None):
    drop_cols = drop_cols or []
    X = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns])
    y = df[target]
    return X, y


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return preprocessor, numeric_features, categorical_features


def get_feature_names(preprocessor, numeric_features, categorical_features):
    names = list(numeric_features)
    try:
        cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features)
        names.extend(cat_names.tolist())
    except Exception:
        names.extend(categorical_features)
    return names
