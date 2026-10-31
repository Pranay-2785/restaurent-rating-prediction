"""Train and evaluate restaurant rating regression models."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_CANDIDATES = (
    "rating",
    "rate",
    "aggregate_rating",
    "aggregate rating",
    "rate (out of 5)",
)
NUMERIC_CANDIDATES = (
    "votes",
    "num of ratings",
    "cost",
    "approx_cost",
    "approx_cost(for two people)",
    "avg cost (two people)",
    "average cost for two",
    "average cost for two people",
    "cost for two",
)
CUISINE_COLUMN_CANDIDATES = ("cuisines type", "cuisines", "cuisine")
CUISINE_FEATURE_PREFIX = "cuisine__"
MAX_RATING = 5.0
DISPLAY_RATING_UPLIFT = 0.3
EXCLUDED_COLUMNS = {
    "url",
    "phone",
    "address",
    "reviews_list",
    "menu_item",
    "dish_liked",
    "restaurant name",
    "local address",
}


def apply_display_uplift(prediction: float) -> float:
    """Apply the disclosed presentation uplift without exceeding the rating scale."""
    return min(MAX_RATING, max(0.0, float(prediction)) + DISPLAY_RATING_UPLIFT)


def build_feature_metadata(features: pd.DataFrame) -> dict[str, Any]:
    """Create the small, CSV-free schema required by the prediction frontend."""
    feature_specs: dict[str, dict[str, Any]] = {}
    for column in features.columns:
        values = features[column]
        if pd.api.types.is_numeric_dtype(values):
            feature_specs[column] = {
                "kind": "numeric",
                "default": float(values.median()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        else:
            options = sorted(values.dropna().astype(str).unique().tolist(), key=str.casefold)
            mode = values.dropna().astype(str).mode()
            feature_specs[column] = {
                "kind": "categorical",
                "options": options,
                "default": str(mode.iloc[0]) if not mode.empty else options[0],
            }

    cuisine_frequency = {
        column: int(features[column].sum())
        for column in features.columns
        if column.startswith(CUISINE_FEATURE_PREFIX)
    }
    return {
        "row_count": len(features),
        "feature_columns": list(features.columns),
        "features": feature_specs,
        "cuisine_frequency": cuisine_frequency,
    }


def parse_rating(value: Any) -> float:
    """Convert common Zomato rating strings to numbers; invalid values become NaN."""
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().upper()
    if text in {"", "NEW", "-", "N/A", "NA", "NONE", "NAN"}:
        return float("nan")
    text = re.sub(r"\s*/\s*5\s*$", "", text)
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _clean_numeric_column(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    return result


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    """Clean a Zomato-like frame and return features, target, and target name."""
    data = _normalize_column_names(frame)
    target_column = next(
        (column for column in TARGET_CANDIDATES if column in data.columns), None
    )
    if target_column is None:
        expected = ", ".join(TARGET_CANDIDATES)
        raise ValueError(f"No rating target column found. Expected one of: {expected}.")

    target = data[target_column].map(parse_rating)
    valid_target = target.notna()
    data = data.loc[valid_target].copy()
    target = target.loc[valid_target].astype(float)

    columns_to_drop = {
        column
        for column in data.columns
        if column in TARGET_CANDIDATES
        or column in EXCLUDED_COLUMNS
        or column.startswith("unnamed:")
    }
    features = data.drop(columns=list(columns_to_drop))

    cuisine_column = next(
        (column for column in CUISINE_COLUMN_CANDIDATES if column in features.columns),
        None,
    )
    if cuisine_column is not None:
        cuisine_values = features.pop(cuisine_column)
        cuisine_lists = cuisine_values.fillna("").astype(str).map(
            lambda value: {
                cuisine.strip()
                for cuisine in value.split(",")
                if cuisine.strip()
            }
        )
        all_cuisines = sorted(
            {cuisine for cuisines in cuisine_lists for cuisine in cuisines},
            key=str.casefold,
        )
        cuisine_features = pd.DataFrame(
            {
                f"{CUISINE_FEATURE_PREFIX}{cuisine}": cuisine_lists.map(
                    lambda cuisines: int(cuisine in cuisines)
                )
                for cuisine in all_cuisines
            },
            index=features.index,
        )
        features = pd.concat([features, cuisine_features], axis=1)

    for column in NUMERIC_CANDIDATES:
        if column in features.columns:
            features[column] = _clean_numeric_column(features[column])

    features = features.dropna(axis=1, how="all")
    if features.empty or features.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleaning.")
    if len(features) < 5:
        raise ValueError(
            "At least five rows with valid ratings are required for a train/test split."
        )
    return features, target, target_column


def print_data_summary(features: pd.DataFrame, target: pd.Series, target_name: str) -> None:
    """Print a concise exploratory summary before model fitting."""
    print(f"Usable rows: {len(features)}")
    print(f"Features: {features.shape[1]} ({', '.join(features.columns)})")
    print(f"Target column: {target_name}")
    print(
        "Target summary: "
        f"mean={target.mean():.3f}, median={target.median():.3f}, "
        f"min={target.min():.3f}, max={target.max():.3f}"
    )
    missing = features.isna().mean().mul(100).sort_values(ascending=False)
    missing = missing[missing > 0].head(10)
    if not missing.empty:
        print("Features with missing values (%):")
        for column, percentage in missing.items():
            print(f"  {column}: {percentage:.1f}%")


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_and_evaluate(
    features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, Pipeline], dict[str, dict[str, float]], str]:
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=max(0.2, 2 / len(features)), random_state=42
    )
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }
    fitted_models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, float]] = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(x_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        fitted_models[name] = pipeline
        metrics[name] = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
            "r2": float(r2_score(y_test, predictions)),
        }

    best_model = min(metrics, key=lambda name: metrics[name]["rmse"])
    return fitted_models, metrics, best_model


def run(data_path: Path, output_dir: Path) -> dict[str, dict[str, float]]:
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Place the CSV there or pass --data."
        )
    raw_data = pd.read_csv(data_path)
    features, target, target_name = prepare_data(raw_data)
    print_data_summary(features, target, target_name)
    fitted_models, metrics, best_model = train_and_evaluate(features, target)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_model.joblib"
    metrics_path = output_dir / "metrics.json"
    schema_path = output_dir / "feature_metadata.json"
    joblib.dump(fitted_models[best_model], model_path)
    metrics_path.write_text(
        json.dumps(
            {"best_model": best_model, "test_metrics": metrics},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    schema_path.write_text(
        json.dumps(build_feature_metadata(features), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nHeld-out test metrics:")
    for name, scores in metrics.items():
        print(
            f"  {name}: MAE={scores['mae']:.3f}, "
            f"RMSE={scores['rmse']:.3f}, R²={scores['r2']:.3f}"
        )
    print(f"Best model by RMSE: {best_model}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved feature metadata: {schema_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/zomato.csv"),
        help="Path to the Zomato CSV (default: data/zomato.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="Directory for the model and metrics (default: artifacts).",
    )
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":
    main()
