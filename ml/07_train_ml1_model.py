from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

INPUT_FILE = PROCESSED_DIR / "ml1_environment_pollution_dataset.csv"

METRICS_FILE = REPORTS_DIR / "final_ml1_metrics.csv"
PREDICTIONS_FILE = REPORTS_DIR / "final_ml1_predictions.csv"
FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "final_ml1_feature_importance.csv"
FEATURE_CONFIG_FILE = REPORTS_DIR / "final_ml1_feature_config.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_TARGET_ROWS = 8
MAX_FEATURE_MISSING_PERCENT = 70


pollutant_targets = [
    "pm25",
    "pm10",
    "so2",
    "co",
    "no2",
    "o3",
    "nh3",
    "no",
    "nox",
]

wind_targets = [
    "wind_u",
    "wind_v",
]

possible_targets = pollutant_targets + wind_targets

weather_features = [
    "temperature",
    "humidity",
    "pressure",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "wind_to_direction",
    "wind_u",
    "wind_v",
    "weather_gap_hours",
]

time_features = [
    "hour",
    "day",
    "month",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

location_features = [
    "latitude",
    "longitude",
]

pollutant_features = [
    "pm25",
    "pm10",
    "so2",
    "co",
    "no2",
    "o3",
    "nh3",
    "no",
    "nox",
]

categorical_features_possible = [
    "station_name",
    "weather_station_name",
]


def clean_column_names(df):
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(".", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    return df


def remove_duplicates_keep_order(items):
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def convert_expected_numeric_columns(df):
    df = df.copy()

    expected_numeric_columns = (
        pollutant_targets
        + wind_targets
        + weather_features
        + location_features
        + time_features
    )

    expected_numeric_columns = remove_duplicates_keep_order(expected_numeric_columns)

    for col in expected_numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def usable_numeric_columns(columns, dataframe, target):
    usable = []

    for col in columns:

        if col not in dataframe.columns:
            continue

        if col == target:
            continue

        missing_percent = dataframe[col].isna().mean() * 100

        if missing_percent > MAX_FEATURE_MISSING_PERCENT:
            print(f"Skipping numeric feature {col}: {missing_percent:.2f}% missing.")
            continue

        if dataframe[col].notna().sum() == 0:
            print(f"Skipping numeric feature {col}: all values missing.")
            continue

        usable.append(col)

    return usable


def usable_categorical_columns(columns, dataframe):
    usable = []

    for col in columns:

        if col not in dataframe.columns:
            continue

        missing_percent = dataframe[col].isna().mean() * 100

        if missing_percent > 95:
            print(f"Skipping categorical feature {col}: {missing_percent:.2f}% missing.")
            continue

        usable.append(col)

    return usable


def get_features_for_target(target, dataframe):

    if target in pollutant_targets:
        numeric_pool = (
            weather_features
            + time_features
            + location_features
            + pollutant_features
        )

        leakage_cols = [
            target,
            "aqi",
        ]

    elif target in wind_targets:
        numeric_pool = (
            [
                "temperature",
                "humidity",
                "pressure",
                "rainfall",
                "weather_gap_hours",
            ]
            + time_features
            + location_features
            + pollutant_features
        )

        leakage_cols = [
            "wind_speed",
            "wind_direction",
            "wind_to_direction",
            "wind_u",
            "wind_v",
        ]

    else:
        numeric_pool = (
            weather_features
            + time_features
            + location_features
            + pollutant_features
        )

        leakage_cols = [
            target,
            "aqi",
        ]

    numeric_pool = remove_duplicates_keep_order(numeric_pool)

    numeric_pool = [
        col for col in numeric_pool
        if col not in leakage_cols
    ]

    numeric_features = usable_numeric_columns(
        columns=numeric_pool,
        dataframe=dataframe,
        target=target
    )

    categorical_features = usable_categorical_columns(
        columns=categorical_features_possible,
        dataframe=dataframe
    )

    return numeric_features, categorical_features


def chronological_train_test_split(model_df, test_size):
    model_df = model_df.copy()

    if "datetime" in model_df.columns:
        model_df = model_df.sort_values("datetime")

    split_index = int(len(model_df) * (1 - test_size))

    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()

    return train_df, test_df


def build_preprocessor(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )

        transformers.append(
            ("num", numeric_transformer, numeric_features)
        )

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        transformers.append(
            ("cat", categorical_transformer, categorical_features)
        )

    if not transformers:
        raise RuntimeError("No usable features available for preprocessing.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


def make_model(numeric_features, categorical_features):
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    regressor = ExtraTreesRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=1
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor)
        ]
    )

    return model


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }

    return metrics


def get_feature_importance(model, numeric_features, categorical_features, target):
    rows = []

    try:
        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["model"]

        feature_names = []

        if numeric_features:
            feature_names.extend(numeric_features)

        if categorical_features:
            cat_transformer = preprocessor.named_transformers_["cat"]
            onehot = cat_transformer.named_steps["onehot"]
            encoded_names = onehot.get_feature_names_out(categorical_features)
            feature_names.extend(encoded_names.tolist())

        importances = regressor.feature_importances_

        for name, importance in zip(feature_names, importances):
            rows.append(
                {
                    "target": target,
                    "feature": name,
                    "importance": float(importance),
                }
            )

    except Exception as error:
        print(f"Could not create feature importance for {target}: {error}")

    return rows


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def train_one_target(df, target):
    print("\n" + "-" * 70)
    print(f"Training target: {target}")
    print("-" * 70)

    if target not in df.columns:
        print(f"Skipping {target}: target column not found.")
        return None

    target_valid_rows = df[target].notna().sum()

    if target_valid_rows < MIN_TARGET_ROWS:
        print(f"Skipping {target}: only {target_valid_rows} valid rows.")
        return None

    numeric_features, categorical_features = get_features_for_target(
        target=target,
        dataframe=df
    )

    print("\nNumeric features used:")
    print(numeric_features)

    print("\nCategorical features used:")
    print(categorical_features)

    selected_features = numeric_features + categorical_features

    if len(selected_features) < 3:
        print(f"Skipping {target}: fewer than 3 usable features.")
        return None

    model_df = df[
        ["datetime", target] + selected_features
    ].copy()

    extra_reference_columns = [
        "station_name",
        "weather_station_name",
        "is_prototype_weather_match",
        "weather_merge_mode",
    ]

    for col in extra_reference_columns:
        if col in df.columns and col not in model_df.columns:
            model_df[col] = df[col]

    model_df = model_df.dropna(subset=[target]).copy()

    if len(model_df) < MIN_TARGET_ROWS:
        print(f"Skipping {target}: not enough rows after target cleanup.")
        return None

    train_df, test_df = chronological_train_test_split(
        model_df=model_df,
        test_size=TEST_SIZE
    )

    if len(train_df) < 3 or len(test_df) < 2:
        print(f"Skipping {target}: train/test split too small.")
        return None

    X_train = train_df[selected_features].copy()
    y_train = train_df[target].copy()

    X_test = test_df[selected_features].copy()
    y_test = test_df[target].copy()

    model = make_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)

    model_file = MODELS_DIR / f"final_ml1_{target}_model.joblib"
    joblib.dump(model, model_file)

    prediction_df = pd.DataFrame()
    prediction_df["datetime"] = test_df["datetime"].values

    if "station_name" in test_df.columns:
        prediction_df["station_name"] = test_df["station_name"].values

    if "weather_station_name" in test_df.columns:
        prediction_df["weather_station_name"] = test_df["weather_station_name"].values

    if "is_prototype_weather_match" in test_df.columns:
        prediction_df["is_prototype_weather_match"] = test_df["is_prototype_weather_match"].values

    if "weather_merge_mode" in test_df.columns:
        prediction_df["weather_merge_mode"] = test_df["weather_merge_mode"].values

    prediction_df["target"] = target
    prediction_df["actual"] = y_test.values
    prediction_df["predicted"] = y_pred
    prediction_df["absolute_error"] = np.abs(y_test.values - y_pred)

    feature_importance_rows = get_feature_importance(
        model=model,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target=target
    )

    result = {
        "target": target,
        "model_file": str(model_file),
        "selected_features": selected_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "metrics": metrics,
        "prediction_df": prediction_df,
        "feature_importance_rows": feature_importance_rows,
    }

    print(f"\nFinished training {target}")
    print("MAE :", metrics["mae"])
    print("RMSE:", metrics["rmse"])
    print("R2  :", metrics["r2"])
    print("Saved model:", model_file)

    return result


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"ML-1 dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df = clean_column_names(df)

    print("ML-1 dataset loaded")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    if "datetime" not in df.columns:
        raise RuntimeError("ML-1 dataset must contain datetime column.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df = convert_expected_numeric_columns(df)

    print("\nDate range:")
    print("Start:", df["datetime"].min())
    print("End  :", df["datetime"].max())

    print("\nChecking possible targets:")
    for target in possible_targets:
        if target in df.columns:
            print(
                target,
                "| valid rows =",
                df[target].notna().sum(),
                "| missing % =",
                round(df[target].isna().mean() * 100, 2)
            )
        else:
            print(target, "| not found")

    all_metrics = []
    all_predictions = []
    all_feature_importance = []
    feature_config = {}

    for target in possible_targets:
        result = train_one_target(df, target)

        if result is None:
            continue

        metric_row = {
            "target": result["target"],
            "model_file": result["model_file"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
            "feature_count": len(result["selected_features"]),
            "numeric_feature_count": len(result["numeric_features"]),
            "categorical_feature_count": len(result["categorical_features"]),
            "mae": result["metrics"]["mae"],
            "rmse": result["metrics"]["rmse"],
            "r2": result["metrics"]["r2"],
        }

        all_metrics.append(metric_row)
        all_predictions.append(result["prediction_df"])
        all_feature_importance.extend(result["feature_importance_rows"])

        feature_config[target] = {
            "target": target,
            "model_file": result["model_file"],
            "selected_features": result["selected_features"],
            "numeric_features": result["numeric_features"],
            "categorical_features": result["categorical_features"],
        }

    metrics_df = pd.DataFrame(all_metrics)

    if metrics_df.empty:
        raise RuntimeError("No model was trained successfully")

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    if all_feature_importance:
        feature_importance_df = pd.DataFrame(all_feature_importance)
    else:
        feature_importance_df = pd.DataFrame(
            columns=[
                "target",
                "feature",
                "importance",
            ]
        )

    metrics_df.to_csv(METRICS_FILE, index=False)
    predictions_df.to_csv(PREDICTIONS_FILE, index=False)
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
    save_json(feature_config, FEATURE_CONFIG_FILE)

    print("\nML-1 training completed successfully")

    print("\nSaved metrics:")
    print(METRICS_FILE)

    print("\nSaved predictions:")
    print(PREDICTIONS_FILE)

    print("\nSaved feature importance:")
    print(FEATURE_IMPORTANCE_FILE)

    print("\nSaved feature config:")
    print(FEATURE_CONFIG_FILE)

    print("\nSaved models folder:")
    print(MODELS_DIR)


if __name__ == "__main__":
    main()





