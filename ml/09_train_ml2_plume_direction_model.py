from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

INPUT_FILE = PROCESSED_DIR / "ml2_plume_training_dataset.csv"

MODEL_FILE = MODELS_DIR / "final_ml2_plume_direction_model.joblib"
METRICS_FILE = REPORTS_DIR / "final_ml2_metrics.csv"
PREDICTIONS_FILE = REPORTS_DIR / "final_ml2_predictions.csv"
FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "final_ml2_feature_importance.csv"
FEATURE_CONFIG_FILE = REPORTS_DIR / "final_ml2_feature_config.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# Model settings

RANDOM_STATE = 42
TEST_SIZE = 0.20

MIN_TOTAL_ROWS = 30
MIN_TRAIN_ROWS = 20
MIN_TEST_ROWS = 5

MAX_NUMERIC_MISSING_PERCENT = 75
MAX_CATEGORICAL_MISSING_PERCENT = 95
MAX_CATEGORICAL_UNIQUE_VALUES = 100

TARGET_COLUMNS = [
    "target_plume_u",
    "target_plume_v",
]

DATETIME_CANDIDATES = [
    "datetime",
    "timestamp",
    "api_time",
    "date_time",
    "time",
    "datetime_ist",
    "datetime_local",
    "datetime_utc",
]

ID_COLUMNS = [
    "id",
    "row_id",
    "record_id",
    "index",
    "unnamed:_0",
    "unnamed_0",
]

REFERENCE_COLUMNS_FOR_OUTPUT = [
    "datetime",
    "target_city",
    "station_name",
    "weather_station_name",
    "source_name",
    "industry_name",
    "source_lat",
    "source_lon",
    "grid_lat",
    "grid_lon",
    "cell_lat",
    "cell_lon",
    "latitude",
    "longitude",
    "pm25",
    "pm10",
    "temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "wind_u",
    "wind_v",
    "ml1_predicted_pm25",
    "ml1_predicted_pm10",
    "ml1_predicted_wind_u",
    "ml1_predicted_wind_v",
    "transport_direction_deg_target",
    "prototype_plume_target_source",
    "ml2_data_source",
]

LEAKAGE_PATTERNS = [
    "target_",
    "actual_plume",
    "observed_plume",
    "true_plume",
    "label_plume",
    "simulation_label",
    "ml2_predicted",
    "angular_error",
    "vector_error",
    "future_",
]

LEAKAGE_COLUMNS = [
    "transport_direction_deg_target",
    "transport_direction_target",
    "transport_direction_deg",
    "prototype_plume_target_source",
]


# Basic helpers

def normalize_column_name(column):
    column = str(column)
    column = column.strip()
    column = column.lower()

    column = column.replace(" ", "_")
    column = column.replace(".", "_")
    column = column.replace("-", "_")
    column = column.replace("/", "_")
    column = column.replace("(", "")
    column = column.replace(")", "")

    return column


def normalize_column_names(df):
    df = df.copy()

    df.columns = [
        normalize_column_name(column)
        for column in df.columns
    ]

    return df


def load_ml2_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"ML-2 training dataset not found:\n{path}\n\n"
            "Run 08_build_ml2_training_dataset.py first."
        )

    df = pd.read_csv(path)
    df = normalize_column_names(df)

    if df.empty:
        raise RuntimeError("ML-2 training dataset is empty.")

    return df


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def find_datetime_column(df):
    for column in DATETIME_CANDIDATES:
        if column in df.columns:
            return column

    return None


def parse_datetime_naive(values):
    parsed = pd.to_datetime(values, errors="coerce")

    try:
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_localize(None)

        return parsed

    except Exception:
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        parsed = parsed.dt.tz_localize(None)

        return parsed


def convert_numeric_like_columns(df):
    df = df.copy()

    for column in df.columns:

        if column in TARGET_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            continue

        if df[column].dtype == "object":
            converted_column = pd.to_numeric(df[column], errors="coerce")
            numeric_ratio = converted_column.notna().mean()

            if numeric_ratio >= 0.80:
                df[column] = converted_column

    return df


# Feature engineering

def add_datetime_features(df):
    df = df.copy()

    datetime_column = find_datetime_column(df)

    if datetime_column is None:
        return df

    if datetime_column != "datetime":
        df["datetime"] = df[datetime_column]

    df["datetime"] = parse_datetime_naive(df["datetime"])
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def looks_like_angle_column(column):
    angle_words = [
        "direction",
        "angle",
        "aspect",
        "bearing",
        "azimuth",
    ]

    for word in angle_words:
        if word in column:
            return True

    return False


def add_angle_encodings(df):
    df = df.copy()

    for column in df.columns:

        if column in TARGET_COLUMNS:
            continue

        if column in LEAKAGE_COLUMNS:
            continue

        if not looks_like_angle_column(column):
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        valid_values = df[column].dropna()

        if valid_values.empty:
            continue

        if valid_values.between(-360, 720).mean() < 0.90:
            continue

        radians = np.deg2rad(df[column])

        sin_column = f"{column}_sin"
        cos_column = f"{column}_cos"

        if sin_column not in df.columns:
            df[sin_column] = np.sin(radians)

        if cos_column not in df.columns:
            df[cos_column] = np.cos(radians)

    return df


def get_first_existing_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column

    return None


def add_spatial_vector_features(df):
    df = df.copy()

    source_lat = get_first_existing_column(
        df,
        [
            "source_lat",
            "industry_lat",
            "emission_lat",
        ]
    )

    source_lon = get_first_existing_column(
        df,
        [
            "source_lon",
            "industry_lon",
            "emission_lon",
        ]
    )

    grid_lat = get_first_existing_column(
        df,
        [
            "grid_lat",
            "cell_lat",
            "receiver_lat",
            "latitude",
        ]
    )

    grid_lon = get_first_existing_column(
        df,
        [
            "grid_lon",
            "cell_lon",
            "receiver_lon",
            "longitude",
        ]
    )

    if all([source_lat, source_lon, grid_lat, grid_lon]):

        if "source_to_grid_lat_delta" not in df.columns:
            df["source_to_grid_lat_delta"] = df[grid_lat] - df[source_lat]

        if "source_to_grid_lon_delta" not in df.columns:
            df["source_to_grid_lon_delta"] = df[grid_lon] - df[source_lon]

        if "source_to_grid_distance_approx" not in df.columns:
            df["source_to_grid_distance_approx"] = np.sqrt(
                df["source_to_grid_lat_delta"] ** 2
                + df["source_to_grid_lon_delta"] ** 2
            )

        if "source_to_grid_angle" not in df.columns:
            df["source_to_grid_angle"] = (
                np.degrees(
                    np.arctan2(
                        df["source_to_grid_lat_delta"],
                        df["source_to_grid_lon_delta"]
                    )
                )
                + 360
            ) % 360

        radians = np.deg2rad(df["source_to_grid_angle"])

        if "source_to_grid_angle_sin" not in df.columns:
            df["source_to_grid_angle_sin"] = np.sin(radians)

        if "source_to_grid_angle_cos" not in df.columns:
            df["source_to_grid_angle_cos"] = np.cos(radians)

    return df


def remove_rows_with_missing_targets(df):
    df = df.copy()

    for target in TARGET_COLUMNS:
        if target not in df.columns:
            raise RuntimeError(
                f"Missing required ML-2 target column: {target}"
            )

        df[target] = pd.to_numeric(df[target], errors="coerce")

    df = df.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)

    if len(df) < MIN_TOTAL_ROWS:
        raise RuntimeError(
            f"Only {len(df)} usable rows found after target cleanup. "
            f"At least {MIN_TOTAL_ROWS} rows are required for ML-2."
        )

    return df


# Feature selection

def is_leakage_column(column):
    if column in TARGET_COLUMNS:
        return True

    if column in ID_COLUMNS:
        return True

    if column in LEAKAGE_COLUMNS:
        return True

    if column.startswith("ml1_predicted_"):
        return False

    for pattern in LEAKAGE_PATTERNS:
        if pattern in column:
            return True

    return False


def get_candidate_feature_columns(df):
    datetime_column = find_datetime_column(df)

    excluded_columns = set(TARGET_COLUMNS + ID_COLUMNS)

    if datetime_column is not None:
        excluded_columns.add(datetime_column)

    candidate_features = []

    for column in df.columns:

        if column in excluded_columns:
            continue

        if is_leakage_column(column):
            continue

        candidate_features.append(column)

    return candidate_features


def remove_bad_feature_columns(df, feature_columns):
    selected_features = []

    removed_feature_report = {
        "too_many_missing_values": [],
        "constant_or_empty": [],
        "too_many_categories": [],
    }

    for column in feature_columns:

        missing_percent = df[column].isna().mean() * 100
        unique_values = df[column].dropna().nunique()

        if unique_values <= 1:
            removed_feature_report["constant_or_empty"].append(column)
            continue

        if pd.api.types.is_numeric_dtype(df[column]):

            if missing_percent > MAX_NUMERIC_MISSING_PERCENT:
                removed_feature_report["too_many_missing_values"].append(
                    {
                        "column": column,
                        "missing_percent": round(float(missing_percent), 4),
                    }
                )
                continue

        else:

            if missing_percent > MAX_CATEGORICAL_MISSING_PERCENT:
                removed_feature_report["too_many_missing_values"].append(
                    {
                        "column": column,
                        "missing_percent": round(float(missing_percent), 4),
                    }
                )
                continue

            if unique_values > MAX_CATEGORICAL_UNIQUE_VALUES:
                removed_feature_report["too_many_categories"].append(
                    {
                        "column": column,
                        "unique_values": int(unique_values),
                    }
                )
                continue

        selected_features.append(column)

    if len(selected_features) < 3:
        raise RuntimeError(
            "Fewer than 3 usable ML-2 features were found."
        )

    return selected_features, removed_feature_report


def split_numeric_and_categorical_features(df, selected_features):
    numeric_features = []
    categorical_features = []

    for column in selected_features:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    return numeric_features, categorical_features


def prepare_ml2_data(df):
    df = df.copy()

    df = convert_numeric_like_columns(df)
    df = add_datetime_features(df)
    df = add_angle_encodings(df)
    df = add_spatial_vector_features(df)
    df = remove_rows_with_missing_targets(df)

    candidate_features = get_candidate_feature_columns(df)

    selected_features, removed_feature_report = remove_bad_feature_columns(
        df,
        candidate_features
    )

    numeric_features, categorical_features = split_numeric_and_categorical_features(
        df,
        selected_features
    )

    X = df[selected_features].copy()
    y = df[TARGET_COLUMNS].copy()

    preparation_report = {
        "total_rows_after_target_cleanup": len(df),
        "selected_features": selected_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "removed_feature_report": removed_feature_report,
    }

    return X, y, df, preparation_report


# Train/test split

def chronological_train_test_split(X, y, prepared_df):
    prepared_df = prepared_df.copy()

    datetime_column = find_datetime_column(prepared_df)

    if datetime_column is not None:
        order = prepared_df[datetime_column].sort_values().index
        X = X.loc[order].reset_index(drop=True)
        y = y.loc[order].reset_index(drop=True)
        prepared_df = prepared_df.loc[order].reset_index(drop=True)

    split_index = int(len(prepared_df) * (1 - TEST_SIZE))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    train_reference = prepared_df.iloc[:split_index].copy()
    test_reference = prepared_df.iloc[split_index:].copy()

    if len(X_train) < MIN_TRAIN_ROWS:
        raise RuntimeError(
            f"Only {len(X_train)} training rows found. "
            f"At least {MIN_TRAIN_ROWS} are required."
        )

    if len(X_test) < MIN_TEST_ROWS:
        raise RuntimeError(
            f"Only {len(X_test)} testing rows found. "
            f"At least {MIN_TEST_ROWS} are required."
        )

    return X_train, X_test, y_train, y_test, train_reference, test_reference


# Model creation

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
        raise RuntimeError("No usable ML-2 features available.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


def build_model(numeric_features, categorical_features):
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    regressor = ExtraTreesRegressor(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor)
        ]
    )

    return pipeline


# Metrics

def calculate_angles_from_uv(u_values, v_values):
    angles = (
        np.degrees(
            np.arctan2(
                v_values,
                u_values
            )
        )
        + 360
    ) % 360

    return angles


def calculate_angular_error(actual_angle, predicted_angle):
    difference = np.abs(actual_angle - predicted_angle)
    difference = np.minimum(difference, 360 - difference)

    return difference


def calculate_metrics(y_test, predictions):
    actual_u = y_test["target_plume_u"].values
    actual_v = y_test["target_plume_v"].values

    predicted_u = predictions[:, 0]
    predicted_v = predictions[:, 1]

    actual_angle = calculate_angles_from_uv(actual_u, actual_v)
    predicted_angle = calculate_angles_from_uv(predicted_u, predicted_v)

    angular_error = calculate_angular_error(
        actual_angle,
        predicted_angle
    )

    vector_error = np.sqrt(
        (actual_u - predicted_u) ** 2
        + (actual_v - predicted_v) ** 2
    )

    metrics = {
        "target_plume_u_mae": round(float(mean_absolute_error(actual_u, predicted_u)), 4),
        "target_plume_u_rmse": round(float(np.sqrt(mean_squared_error(actual_u, predicted_u))), 4),
        "target_plume_u_r2": round(float(r2_score(actual_u, predicted_u)), 4),

        "target_plume_v_mae": round(float(mean_absolute_error(actual_v, predicted_v)), 4),
        "target_plume_v_rmse": round(float(np.sqrt(mean_squared_error(actual_v, predicted_v))), 4),
        "target_plume_v_r2": round(float(r2_score(actual_v, predicted_v)), 4),

        "mean_vector_error": round(float(np.mean(vector_error)), 4),
        "median_vector_error": round(float(np.median(vector_error)), 4),

        "mean_angular_error_degrees": round(float(np.mean(angular_error)), 4),
        "median_angular_error_degrees": round(float(np.median(angular_error)), 4),
    }

    return metrics, actual_angle, predicted_angle, angular_error, vector_error


# Outputs

def get_reference_output(test_reference):
    output = pd.DataFrame()

    for column in REFERENCE_COLUMNS_FOR_OUTPUT:
        if column in test_reference.columns:
            output[column] = test_reference[column].values

    return output


def create_predictions_dataframe(test_reference, y_test, predictions):
    prediction_df = get_reference_output(test_reference)

    actual_u = y_test["target_plume_u"].values
    actual_v = y_test["target_plume_v"].values

    predicted_u = predictions[:, 0]
    predicted_v = predictions[:, 1]

    actual_angle = calculate_angles_from_uv(actual_u, actual_v)
    predicted_angle = calculate_angles_from_uv(predicted_u, predicted_v)

    angular_error = calculate_angular_error(
        actual_angle,
        predicted_angle
    )

    vector_error = np.sqrt(
        (actual_u - predicted_u) ** 2
        + (actual_v - predicted_v) ** 2
    )

    prediction_df["actual_target_plume_u"] = actual_u
    prediction_df["actual_target_plume_v"] = actual_v

    prediction_df["predicted_target_plume_u"] = predicted_u
    prediction_df["predicted_target_plume_v"] = predicted_v

    prediction_df["actual_plume_angle"] = actual_angle
    prediction_df["predicted_plume_angle"] = predicted_angle

    prediction_df["angular_error"] = angular_error
    prediction_df["vector_error"] = vector_error

    return prediction_df


def get_feature_names_from_preprocessor(model, numeric_features, categorical_features):
    feature_names = []

    preprocessor = model.named_steps["preprocessor"]

    if numeric_features:
        feature_names.extend(numeric_features)

    if categorical_features:
        try:
            categorical_transformer = preprocessor.named_transformers_["cat"]
            onehot = categorical_transformer.named_steps["onehot"]
            encoded_names = onehot.get_feature_names_out(categorical_features)
            feature_names.extend(encoded_names.tolist())

        except Exception:
            feature_names.extend(categorical_features)

    return feature_names


def create_feature_importance_dataframe(model, numeric_features, categorical_features):
    regressor = model.named_steps["model"]

    feature_names = get_feature_names_from_preprocessor(
        model,
        numeric_features,
        categorical_features
    )

    importances = regressor.feature_importances_

    rows = []

    for feature, importance in zip(feature_names, importances):
        rows.append(
            {
                "feature": feature,
                "importance": float(importance),
            }
        )

    importance_df = pd.DataFrame(rows)

    if not importance_df.empty:
        importance_df = importance_df.sort_values(
            "importance",
            ascending=False
        ).reset_index(drop=True)

    return importance_df


def save_outputs(model, metrics, predictions_df, feature_importance_df, preparation_report):
    artifact = {
        "pipeline": model,
        "target_columns": TARGET_COLUMNS,
        "feature_columns": preparation_report["selected_features"],
        "numeric_features": preparation_report["numeric_features"],
        "categorical_features": preparation_report["categorical_features"],
        "metrics": metrics,
    }

    joblib.dump(artifact, MODEL_FILE)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(METRICS_FILE, index=False)

    predictions_df.to_csv(PREDICTIONS_FILE, index=False)
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    feature_config = {
        "target_columns": TARGET_COLUMNS,
        "selected_features": preparation_report["selected_features"],
        "numeric_features": preparation_report["numeric_features"],
        "categorical_features": preparation_report["categorical_features"],
        "removed_feature_report": preparation_report["removed_feature_report"],
        "model_file": str(MODEL_FILE),
        "metrics_file": str(METRICS_FILE),
        "predictions_file": str(PREDICTIONS_FILE),
        "feature_importance_file": str(FEATURE_IMPORTANCE_FILE),
    }

    save_json(feature_config, FEATURE_CONFIG_FILE)

    print("\nML-2 training completed successfully.")

    print("\nSaved ML-2 model:")
    print(MODEL_FILE)

    print("\nSaved metrics:")
    print(METRICS_FILE)

    print("\nSaved predictions:")
    print(PREDICTIONS_FILE)

    print("\nSaved feature importance:")
    print(FEATURE_IMPORTANCE_FILE)

    print("\nSaved feature config:")
    print(FEATURE_CONFIG_FILE)


# Run

def main():
    print("Project root:")
    print(PROJECT_ROOT)

    print("\nLoading ML-2 training dataset:")
    print(INPUT_FILE)

    df = load_ml2_dataset(INPUT_FILE)

    print("\nML-2 dataset loaded.")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nPreparing ML-2 dataset...")

    X, y, prepared_df, preparation_report = prepare_ml2_data(df)

    print("\nSelected features:")
    print(preparation_report["selected_features"])

    print("\nNumeric features:")
    print(preparation_report["numeric_features"])

    print("\nCategorical features:")
    print(preparation_report["categorical_features"])

    X_train, X_test, y_train, y_test, train_reference, test_reference = chronological_train_test_split(
        X,
        y,
        prepared_df
    )

    print("\nTrain rows:", len(X_train))
    print("Test rows :", len(X_test))

    model = build_model(
        preparation_report["numeric_features"],
        preparation_report["categorical_features"]
    )

    print("\nTraining ML-2 model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics, actual_angle, predicted_angle, angular_error, vector_error = calculate_metrics(
        y_test,
        predictions
    )

    print("\nML-2 metrics:")
    for key, value in metrics.items():
        print(key, ":", value)

    predictions_df = create_predictions_dataframe(
        test_reference,
        y_test,
        predictions
    )

    feature_importance_df = create_feature_importance_dataframe(
        model,
        preparation_report["numeric_features"],
        preparation_report["categorical_features"]
    )

    save_outputs(
        model,
        metrics,
        predictions_df,
        feature_importance_df,
        preparation_report
    )


if __name__ == "__main__":
    main()












    







    





