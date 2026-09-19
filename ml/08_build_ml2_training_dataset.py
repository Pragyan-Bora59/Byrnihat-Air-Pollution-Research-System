from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

ML1_INPUT_FILE = PROCESSED_DIR / "ml1_environment_pollution_dataset.csv"
SIMULATION_LABEL_FILE = PROCESSED_DIR / "plume_simulation_labels.csv"

ML1_FEATURE_CONFIG_FILE = REPORTS_DIR / "final_ml1_feature_config.json"

OUTPUT_FILE = PROCESSED_DIR / "ml2_plume_training_dataset.csv"
BUILD_REPORT_FILE = REPORTS_DIR / "ml2_dataset_build_report.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_PLUME_TARGETS = [
    "target_plume_u",
    "target_plume_v",
]

DATETIME_CANDIDATES = [
    "datetime",
    "api_time",
    "timestamp",
    "date_time",
    "time",
    "datetime_ist",
    "datetime_local",
    "datetime_utc",
]


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


def load_csv_file(path, description):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")

    df = pd.read_csv(path)
    df = normalize_column_names(df)

    if df.empty:
        raise RuntimeError(f"{description} is empty: {path}")

    return df


def load_json_file(path, description):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not data:
        raise RuntimeError(f"{description} is empty.")

    return data


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


def ensure_datetime(df, file_name):
    df = df.copy()

    datetime_column = find_datetime_column(df)

    if datetime_column is None:
        raise RuntimeError(f"{file_name} must contain a datetime column.")

    if datetime_column != "datetime":
        df["datetime"] = df[datetime_column]

    df["datetime"] = parse_datetime_naive(df["datetime"])
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def fix_station_column(df):
    df = df.copy()

    if "station_name" not in df.columns and "target_city" in df.columns:
        df["station_name"] = df["target_city"]

    if "target_city" not in df.columns and "station_name" in df.columns:
        df["target_city"] = df["station_name"]

    if "station_name" in df.columns:
        df["station_name"] = df["station_name"].astype(str).str.strip()

    if "target_city" in df.columns:
        df["target_city"] = df["target_city"].astype(str).str.strip()

    return df


def convert_numeric_like_columns(df):
    df = df.copy()

    for column in df.columns:

        if column in REQUIRED_PLUME_TARGETS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            continue

        if df[column].dtype == "object":
            converted_column = pd.to_numeric(df[column], errors="coerce")
            numeric_ratio = converted_column.notna().mean()

            if numeric_ratio >= 0.80:
                df[column] = converted_column

    return df


def create_plume_targets_from_direction(df):
    df = df.copy()

    if "target_plume_u" in df.columns and "target_plume_v" in df.columns:
        return df

    direction_column = None

    possible_direction_columns = [
        "transport_direction_deg_target",
        "transport_direction_target",
        "transport_direction_deg",
        "wind_to_direction_proxy",
        "met_wind_to_direction",
        "wind_to_direction",
        "met_wind_direction",
        "wind_direction",
    ]

    for column in possible_direction_columns:
        if column in df.columns:
            direction_column = column
            break

    if direction_column is None:
        return df

    df[direction_column] = pd.to_numeric(
        df[direction_column],
        errors="coerce"
    )

    if "met_wind_speed" in df.columns:
        wind_speed = pd.to_numeric(df["met_wind_speed"], errors="coerce")
    elif "wind_speed" in df.columns:
        wind_speed = pd.to_numeric(df["wind_speed"], errors="coerce")
    else:
        wind_speed = pd.Series(1, index=df.index)

    if "pm25" in df.columns:
        pollution_strength = pd.to_numeric(df["pm25"], errors="coerce")
    elif "pm10" in df.columns:
        pollution_strength = pd.to_numeric(df["pm10"], errors="coerce")
    else:
        pollution_strength = pd.Series(1, index=df.index)

    if "met_rainfall" in df.columns:
        rainfall = pd.to_numeric(df["met_rainfall"], errors="coerce")
    elif "rainfall" in df.columns:
        rainfall = pd.to_numeric(df["rainfall"], errors="coerce")
    else:
        rainfall = pd.Series(0, index=df.index)

    wind_speed = wind_speed.fillna(1)

    pollution_median = pollution_strength.median()

    if pd.isna(pollution_median) or pollution_median == 0:
        pollution_median = 1

    pollution_strength = pollution_strength.fillna(pollution_median)
    rainfall = rainfall.fillna(0)

    pollution_scale = pollution_strength / pollution_median
    rain_factor = 1 / (1 + rainfall)

    plume_strength = wind_speed * pollution_scale * rain_factor

    plume_strength = plume_strength.replace([np.inf, -np.inf], np.nan)

    plume_strength_median = plume_strength.median()

    if pd.isna(plume_strength_median) or plume_strength_median == 0:
        plume_strength_median = 1

    plume_strength = plume_strength.fillna(plume_strength_median)

    angle_radians = np.deg2rad(df[direction_column])

    df["target_plume_u"] = plume_strength * np.cos(angle_radians)
    df["target_plume_v"] = plume_strength * np.sin(angle_radians)

    df["prototype_plume_target_source"] = direction_column

    print("Created target_plume_u and target_plume_v from:")
    print(direction_column)

    return df


def validate_plume_targets(df):
    df = df.copy()

    missing_targets = []

    for target in REQUIRED_PLUME_TARGETS:
        if target not in df.columns:
            missing_targets.append(target)

    if missing_targets:
        raise RuntimeError(
            "ML-2 dataset is missing required target columns:\n"
            f"{missing_targets}\n\n"
            "The dataset must contain:\n"
            "target_plume_u\n"
            "target_plume_v"
        )

    for target in REQUIRED_PLUME_TARGETS:
        df[target] = pd.to_numeric(df[target], errors="coerce")

    df = df.dropna(subset=REQUIRED_PLUME_TARGETS).reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            "ML-2 dataset became empty after removing missing plume targets."
        )

    return df


def find_prototype_ml_train_file():
    expected_file = (
        DATA_DIR
        / "prototype"
        / "byrnihat_missing_source_proxy_datasets"
        / "ml_ready"
        / "prototype_ml_train.csv"
    )

    if expected_file.exists():
        return expected_file

    files = sorted(
        DATA_DIR.rglob("prototype_ml_train.csv")
    )

    if files:
        return files[0]

    return None


def resolve_model_path(model_file):
    model_path = Path(model_file)

    if model_path.exists():
        return model_path

    possible_path = PROJECT_ROOT / model_file

    if possible_path.exists():
        return possible_path

    possible_path = MODELS_DIR / Path(model_file).name

    if possible_path.exists():
        return possible_path

    raise FileNotFoundError(
        f"Saved ML-1 model not found: {model_file}"
    )


def get_selected_features_from_config(config):
    selected_features = config.get("selected_features", [])

    if selected_features:
        return selected_features

    numeric_features = config.get("numeric_features", [])
    categorical_features = config.get("categorical_features", [])

    selected_features = numeric_features + categorical_features

    return selected_features


def create_ml1_prediction_features(ml1_df, feature_config):
    prediction_df = ml1_df.copy()

    report = {
        "ml1_targets_used": [],
        "ml1_targets_skipped": [],
        "missing_features_created_as_nan": {},
    }

    for target in feature_config:

        config = feature_config[target]

        model_file = config.get("model_file")
        selected_features = get_selected_features_from_config(config)

        if model_file is None or not selected_features:
            report["ml1_targets_skipped"].append(
                {
                    "target": target,
                    "reason": "model_file or selected_features missing",
                }
            )
            continue

        try:
            model_path = resolve_model_path(model_file)
            model = joblib.load(model_path)

        except Exception as error:
            report["ml1_targets_skipped"].append(
                {
                    "target": target,
                    "reason": str(error),
                }
            )
            continue

        missing_features = []

        for feature in selected_features:
            if feature not in prediction_df.columns:
                prediction_df[feature] = np.nan
                missing_features.append(feature)

        if missing_features:
            report["missing_features_created_as_nan"][target] = missing_features

        X = prediction_df[selected_features].copy()

        predicted_column = f"ml1_predicted_{target}"

        try:
            prediction_df[predicted_column] = model.predict(X)

        except Exception as error:
            report["ml1_targets_skipped"].append(
                {
                    "target": target,
                    "reason": f"prediction failed: {error}",
                }
            )
            continue

        report["ml1_targets_used"].append(
            {
                "target": target,
                "model_file": str(model_path),
                "prediction_column": predicted_column,
                "feature_count": len(selected_features),
            }
        )

    return prediction_df, report


def add_predicted_wind_features(df):
    df = df.copy()

    u_col = "ml1_predicted_wind_u"
    v_col = "ml1_predicted_wind_v"

    if u_col in df.columns and v_col in df.columns:

        df["ml1_predicted_wind_speed"] = np.sqrt(
            df[u_col] ** 2
            + df[v_col] ** 2
        )

        df["ml1_predicted_wind_angle"] = (
            np.degrees(
                np.arctan2(
                    df[v_col],
                    df[u_col]
                )
            )
            + 360
        ) % 360

    return df


def load_ml1_prediction_dataset():
    print("Loading ML-1 dataset:")
    print(ML1_INPUT_FILE)

    ml1_df = load_csv_file(
        ML1_INPUT_FILE,
        "ML-1 environment pollution dataset"
    )

    ml1_df = ensure_datetime(ml1_df, "ML-1 dataset")
    ml1_df = fix_station_column(ml1_df)
    ml1_df = convert_numeric_like_columns(ml1_df)

    print("ML-1 rows:")
    print(len(ml1_df))

    print("Loading ML-1 feature config:")
    print(ML1_FEATURE_CONFIG_FILE)

    feature_config = load_json_file(
        ML1_FEATURE_CONFIG_FILE,
        "ML-1 feature config"
    )

    ml1_prediction_df, prediction_report = create_ml1_prediction_features(
        ml1_df,
        feature_config
    )

    ml1_prediction_df = add_predicted_wind_features(ml1_prediction_df)

    ml1_prediction_columns = []

    for column in ml1_prediction_df.columns:
        if column.startswith("ml1_predicted_"):
            ml1_prediction_columns.append(column)

    if not ml1_prediction_columns:
        raise RuntimeError(
            "No ML-1 prediction columns were created. "
            "Run 07_train_ml1_model.py first."
        )

    print("ML-1 prediction columns created:")
    print(ml1_prediction_columns)

    return ml1_prediction_df, prediction_report


def get_merge_keys(left_df, right_df):
    merge_keys = []

    if "datetime" in left_df.columns and "datetime" in right_df.columns:
        merge_keys.append("datetime")

    if "station_name" in left_df.columns and "station_name" in right_df.columns:
        merge_keys.append("station_name")

    if "datetime" not in merge_keys:
        raise RuntimeError(
            "Cannot merge because datetime is missing from one dataset."
        )

    return merge_keys


def prepare_ml1_for_merge(ml1_prediction_df, merge_keys, base_columns):
    columns_to_keep = []

    for column in merge_keys:
        if column in ml1_prediction_df.columns:
            columns_to_keep.append(column)

    for column in ml1_prediction_df.columns:
        if column.startswith("ml1_predicted_"):
            if column not in columns_to_keep:
                columns_to_keep.append(column)

    reference_columns = [
        "pm25",
        "pm10",
        "aqi",
        "so2",
        "co",
        "no2",
        "o3",
        "nh3",
        "no",
        "nox",
        "temperature",
        "humidity",
        "pressure",
        "rainfall",
        "wind_speed",
        "wind_direction",
        "wind_to_direction",
        "wind_u",
        "wind_v",
        "weather_station_name",
        "weather_gap_hours",
        "weather_merge_mode",
        "is_prototype_weather_match",
    ]

    for column in reference_columns:
        if column in ml1_prediction_df.columns:
            if column not in columns_to_keep and column not in base_columns:
                columns_to_keep.append(column)

    ml1_for_merge = ml1_prediction_df[columns_to_keep].copy()

    if "datetime" in merge_keys:
        ml1_for_merge["datetime"] = parse_datetime_naive(
            ml1_for_merge["datetime"]
        )

    if "station_name" in merge_keys:
        ml1_for_merge["station_name"] = (
            ml1_for_merge["station_name"]
            .astype(str)
            .str.strip()
        )

    ml1_for_merge = ml1_for_merge.drop_duplicates(subset=merge_keys)

    return ml1_for_merge


def attach_ml1_predictions(base_df, ml1_prediction_df):
    base_df = base_df.copy()
    ml1_prediction_df = ml1_prediction_df.copy()

    base_df = ensure_datetime(base_df, "ML-2 base dataset")
    ml1_prediction_df = ensure_datetime(
        ml1_prediction_df,
        "ML-1 prediction dataset"
    )

    base_df = fix_station_column(base_df)
    ml1_prediction_df = fix_station_column(ml1_prediction_df)

    merge_keys = get_merge_keys(base_df, ml1_prediction_df)

    ml1_for_merge = prepare_ml1_for_merge(
        ml1_prediction_df,
        merge_keys,
        base_df.columns.tolist()
    )

    merged_df = base_df.merge(
        ml1_for_merge,
        on=merge_keys,
        how="left"
    )

    return merged_df, merge_keys


def load_simulation_label_file():
    print("Loading simulation plume label file:")
    print(SIMULATION_LABEL_FILE)

    df = load_csv_file(
        SIMULATION_LABEL_FILE,
        "Simulation plume label dataset"
    )

    df = ensure_datetime(df, "Simulation plume label dataset")
    df = fix_station_column(df)
    df = convert_numeric_like_columns(df)

    df = create_plume_targets_from_direction(df)

    df = validate_plume_targets(df)

    df["ml2_data_source"] = "simulation_label_file"

    return df


def load_prototype_ml_train_file():
    prototype_file = find_prototype_ml_train_file()

    if prototype_file is None:
        raise FileNotFoundError(
            "Could not find prototype_ml_train.csv inside data folder."
        )

    print("Using prototype ML-2 training file:")
    print(prototype_file)

    df = pd.read_csv(prototype_file)
    df = normalize_column_names(df)

    print("Prototype columns:")
    print(df.columns.tolist())

    df = ensure_datetime(df, "prototype_ml_train.csv")
    df = fix_station_column(df)
    df = convert_numeric_like_columns(df)

    df = create_plume_targets_from_direction(df)

    df = validate_plume_targets(df)

    df["ml2_data_source"] = "prototype_ml_train_file"

    return df


def order_output_columns(df):
    df = df.copy()

    important_columns = [
        "datetime",
        "target_city",
        "station_name",

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
        "aqi",
        "so2",
        "co",
        "no2",
        "o3",
        "nh3",
        "no",
        "nox",

        "temperature",
        "humidity",
        "pressure",
        "rainfall",
        "wind_speed",
        "wind_direction",
        "wind_to_direction",
        "wind_u",
        "wind_v",

        "ml1_predicted_pm25",
        "ml1_predicted_pm10",
        "ml1_predicted_so2",
        "ml1_predicted_co",
        "ml1_predicted_no2",
        "ml1_predicted_o3",
        "ml1_predicted_nh3",
        "ml1_predicted_no",
        "ml1_predicted_nox",
        "ml1_predicted_wind_u",
        "ml1_predicted_wind_v",
        "ml1_predicted_wind_speed",
        "ml1_predicted_wind_angle",

        "transport_direction_deg_target",
        "target_plume_u",
        "target_plume_v",
        "prototype_plume_target_source",
        "ml2_data_source",
    ]

    selected_columns = []

    for column in important_columns:
        if column in df.columns:
            selected_columns.append(column)

    remaining_columns = []

    for column in df.columns:
        if column not in selected_columns:
            remaining_columns.append(column)

    df = df[selected_columns + remaining_columns]

    return df


def save_outputs(ml2_df, merge_keys, ml1_prediction_report):
    ml2_df = order_output_columns(ml2_df)

    ml2_df.to_csv(OUTPUT_FILE, index=False)

    build_report = {
        "script_name": "08_build_ml2_training_dataset.py",
        "purpose": "Build ML-2 plume direction training dataset",
        "ml1_input_file": str(ML1_INPUT_FILE),
        "simulation_label_file": str(SIMULATION_LABEL_FILE),
        "ml1_feature_config_file": str(ML1_FEATURE_CONFIG_FILE),
        "output_file": str(OUTPUT_FILE),
        "merge_keys": merge_keys,
        "final_rows": len(ml2_df),
        "final_columns": ml2_df.columns.tolist(),
        "required_targets": REQUIRED_PLUME_TARGETS,
        "ml1_prediction_report": ml1_prediction_report,
    }

    save_json(build_report, BUILD_REPORT_FILE)

    print("ML-2 training dataset created successfully.")

    print("\nSaved ML-2 dataset:")
    print(OUTPUT_FILE)

    print("\nSaved build report:")
    print(BUILD_REPORT_FILE)

    print("\nFinal rows:")
    print(len(ml2_df))

    print("\nFinal columns:")
    print(ml2_df.columns.tolist())

    print("\nTarget columns check:")
    for target in REQUIRED_PLUME_TARGETS:
        print(target, "=", target in ml2_df.columns)

    print("\nFirst 5 rows:")
    print(ml2_df.head())


def main():
    print("Building ML-2 training dataset.")

    ml1_prediction_df, ml1_prediction_report = load_ml1_prediction_dataset()

    if SIMULATION_LABEL_FILE.exists():
        print("\nUsing simulation label file.")
        base_df = load_simulation_label_file()
    else:
        print("\nSimulation label file was not found.")
        print("Using prototype_ml_train.csv.")
        base_df = load_prototype_ml_train_file()

    ml2_df, merge_keys = attach_ml1_predictions(
        base_df,
        ml1_prediction_df
    )

    ml2_df = validate_plume_targets(ml2_df)

    if "ml2_data_source" not in ml2_df.columns:
        ml2_df["ml2_data_source"] = "ml2_training_dataset"

    save_outputs(
        ml2_df,
        merge_keys,
        ml1_prediction_report
    )


if __name__ == "__main__":
    main()