from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
MAPS_DIR = PROJECT_ROOT / "outputs" / "maps"

ML1_BASE_DATA_FILE = PROCESSED_DIR / "ml1_environment_pollution_dataset.csv"
ML2_BASE_DATA_FILE = PROCESSED_DIR / "ml2_plume_training_dataset.csv"

ML1_FEATURE_CONFIG_FILE = REPORTS_DIR / "final_ml1_feature_config.json"
ML2_FEATURE_CONFIG_FILE = REPORTS_DIR / "final_ml2_feature_config.json"

ML2_MODEL_FILE = MODELS_DIR / "final_ml2_plume_direction_model.joblib"

FINAL_PREDICTION_FILE = REPORTS_DIR / "final_prediction_output.csv"
FINAL_GRAPH_DATA_FILE = REPORTS_DIR / "final_dashboard_graph_data.json"
PIPELINE_REPORT_FILE = REPORTS_DIR / "final_prediction_pipeline_report.json"

ML_PLUME_SIMULATION_JSON_FILE = MAPS_DIR / "ml_predicted_plume_simulation_data.json"
ML_PLUME_SIMULATION_JS_FILE = MAPS_DIR / "ml_predicted_plume_simulation_data.js"
ML_PLUME_FRAMES_CSV_FILE = MAPS_DIR / "ml_predicted_plume_frames.csv"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

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

MERGE_KEY_CANDIDATES = [
    "datetime",
    "station_name",
]

REFERENCE_COLUMNS = [
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
]

POLLUTION_COLUMNS = [
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

WEATHER_COLUMNS = [
    "temperature",
    "humidity",
    "pressure",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "wind_to_direction",
    "wind_u",
    "wind_v",
]

PLUME_FRAME_COUNT = 30
PARTICLES_PER_FRAME = 35
MAX_RECORDS_FOR_HTML = 120

PLUME_STEP_SCALE = 0.01
PARTICLE_SPREAD_SCALE = 0.002

DEFAULT_SOURCE_LAT = 26.065
DEFAULT_SOURCE_LON = 91.875
DEFAULT_SOURCE_NAME = "Prototype Byrnihat industrial source"

MIN_FINAL_ROWS = 1


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
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    df = pd.read_csv(path)
    df = normalize_column_names(df)

    if df.empty:
        raise RuntimeError(f"{description} is empty.")

    return df


def load_json_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not data:
        raise RuntimeError(f"{description} is empty.")

    return data


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_js_variable(variable_name, data, path):
    json_text = json.dumps(data, indent=2)

    js_text = (
        f"window.{variable_name} = "
        f"{json_text};\n"
    )

    with open(path, "w", encoding="utf-8") as file:
        file.write(js_text)


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


def prepare_datetime(df):
    df = df.copy()

    datetime_column = find_datetime_column(df)

    if datetime_column is None:
        return df

    if datetime_column != "datetime":
        df["datetime"] = df[datetime_column]

    df["datetime"] = parse_datetime_naive(df["datetime"])
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def convert_numeric_like_columns(df):
    df = df.copy()

    for column in df.columns:

        if df[column].dtype == "object":
            converted_column = pd.to_numeric(df[column], errors="coerce")
            numeric_ratio = converted_column.notna().mean()

            if numeric_ratio >= 0.80:
                df[column] = converted_column

    return df


def safe_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


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
        f"Could not find saved model:\n{model_file}"
    )


def get_selected_features_from_config(config):
    selected_features = config.get("selected_features", [])

    if selected_features:
        return selected_features

    numeric_features = config.get("numeric_features", [])
    categorical_features = config.get("categorical_features", [])

    selected_features = numeric_features + categorical_features

    return selected_features


def create_ml1_prediction_features(ml1_df, ml1_feature_config):
    prediction_df = ml1_df.copy()

    report = {
        "ml1_targets_used": [],
        "ml1_targets_skipped": [],
        "missing_features_created_as_nan": {},
    }

    for target in ml1_feature_config:

        config = ml1_feature_config[target]

        model_file = config.get("model_file")
        selected_features = get_selected_features_from_config(config)

        if model_file is None or not selected_features:
            report["ml1_targets_skipped"].append(
                {
                    "target": target,
                    "reason": "Missing model_file or selected_features.",
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
                missing_features.append(feature)
                prediction_df[feature] = np.nan

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
                    "reason": f"Prediction failed: {error}",
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


def add_ml1_predicted_wind_features(df):
    df = df.copy()

    u_column = "ml1_predicted_wind_u"
    v_column = "ml1_predicted_wind_v"

    if u_column in df.columns and v_column in df.columns:

        df["ml1_predicted_wind_speed"] = np.sqrt(
            df[u_column] ** 2
            + df[v_column] ** 2
        )

        df["ml1_predicted_wind_angle"] = (
            np.degrees(
                np.arctan2(
                    df[v_column],
                    df[u_column]
                )
            )
            + 360
        ) % 360

    return df


def get_merge_keys(left_df, right_df):
    merge_keys = []

    for key in MERGE_KEY_CANDIDATES:
        if key in left_df.columns and key in right_df.columns:
            merge_keys.append(key)

    if "datetime" not in merge_keys:
        raise RuntimeError(
            "Cannot merge ML-1 predictions with ML-2 data because "
            "datetime is missing from one of the files."
        )

    return merge_keys


def prepare_ml1_for_merge(ml1_prediction_df, merge_keys):
    ml1_prediction_df = ml1_prediction_df.copy()

    ml1_prediction_columns = []

    for column in ml1_prediction_df.columns:
        if column.startswith("ml1_predicted_"):
            ml1_prediction_columns.append(column)

    if not ml1_prediction_columns:
        raise RuntimeError("No ML-1 prediction columns were created.")

    columns_to_merge = merge_keys + ml1_prediction_columns

    columns_to_merge = [
        column for column in columns_to_merge
        if column in ml1_prediction_df.columns
    ]

    ml1_for_merge = ml1_prediction_df[columns_to_merge].copy()

    if merge_keys == ["datetime"]:
        numeric_columns = []

        for column in ml1_for_merge.columns:
            if column != "datetime" and pd.api.types.is_numeric_dtype(ml1_for_merge[column]):
                numeric_columns.append(column)

        ml1_for_merge = (
            ml1_for_merge
            .groupby("datetime", as_index=False)[numeric_columns]
            .mean()
        )

    else:
        ml1_for_merge = ml1_for_merge.drop_duplicates(subset=merge_keys)

    return ml1_for_merge, ml1_prediction_columns


def merge_ml1_predictions_with_ml2_base(ml2_base_df, ml1_prediction_df):

    ml2_base_df = ml2_base_df.copy()
    ml1_prediction_df = ml1_prediction_df.copy()

    ml2_base_df = prepare_datetime(ml2_base_df)
    ml1_prediction_df = prepare_datetime(ml1_prediction_df)

    ml1_prediction_columns = []

    for column in ml1_prediction_df.columns:
        if column.startswith("ml1_predicted_"):
            ml1_prediction_columns.append(column)

    if not ml1_prediction_columns:
        raise RuntimeError("No ML-1 prediction columns were created.")

    if "station_name" in ml2_base_df.columns:
        ml2_base_df["station_name"] = ml2_base_df["station_name"].astype(str).str.strip()

    if "station_name" in ml1_prediction_df.columns:
        ml1_prediction_df["station_name"] = ml1_prediction_df["station_name"].astype(str).str.strip()

    merge_keys = get_merge_keys(ml2_base_df, ml1_prediction_df)

    columns_to_merge = merge_keys + ml1_prediction_columns

    columns_to_merge = [
        column for column in columns_to_merge
        if column in ml1_prediction_df.columns
    ]

    ml1_for_merge = ml1_prediction_df[columns_to_merge].copy()

    exact_merged_df = ml2_base_df.merge(
        ml1_for_merge,
        on=merge_keys,
        how="left",
        suffixes=("", "_new")
    )

    missing_after_exact = exact_merged_df[ml1_prediction_columns].isna().mean().mean()

    if missing_after_exact < 0.50:
        print("Used exact merge for ML-1 predictions.")
        return exact_merged_df, merge_keys, ml1_prediction_columns

    print("Exact merge gave too many missing ML-1 predictions.")
    print("Using nearest-time merge for prototype/final compatibility.")

    nearest_parts = []

    if "station_name" in ml2_base_df.columns and "station_name" in ml1_prediction_df.columns:

        for station_name, base_station_df in ml2_base_df.groupby("station_name", dropna=False):

            station_string = str(station_name).strip()

            left_part = base_station_df.sort_values("datetime").copy()

            right_part = ml1_prediction_df[
                ml1_prediction_df["station_name"].astype(str).str.strip() == station_string
            ].copy()

            if right_part.empty:
                right_part = ml1_prediction_df.copy()

            right_part = right_part.sort_values("datetime")

            right_keep_columns = ["datetime"] + ml1_prediction_columns

            right_part = right_part[right_keep_columns].copy()

            merged_part = pd.merge_asof(
                left_part,
                right_part,
                on="datetime",
                direction="nearest",
                tolerance=pd.Timedelta("24h"),
                suffixes=("", "_new")
            )

            nearest_parts.append(merged_part)

        merged_df = pd.concat(nearest_parts, ignore_index=True)

        merge_keys = ["datetime", "station_name_nearest_time"]

    else:

        left_df = ml2_base_df.sort_values("datetime").copy()
        right_df = ml1_prediction_df.sort_values("datetime").copy()

        right_keep_columns = ["datetime"] + ml1_prediction_columns
        right_df = right_df[right_keep_columns].copy()

        merged_df = pd.merge_asof(
            left_df,
            right_df,
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta("24h"),
            suffixes=("", "_new")
        )

        merge_keys = ["datetime_nearest_time"]

    for column in ml1_prediction_columns:

        new_column = f"{column}_new"

        if new_column in merged_df.columns:

            if column in merged_df.columns:
                merged_df[column] = merged_df[column].combine_first(
                    merged_df[new_column]
                )
            else:
                merged_df[column] = merged_df[new_column]

            merged_df = merged_df.drop(columns=[new_column])

    missing_after_nearest = merged_df[ml1_prediction_columns].isna().mean().mean()

    print("Average missing ML-1 prediction percent after nearest merge:")
    print(round(float(missing_after_nearest * 100), 2))

    return merged_df, merge_keys, ml1_prediction_columns


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


def prepare_ml2_prediction_features(df, ml2_feature_columns):
    df = df.copy()

    df = convert_numeric_like_columns(df)
    df = add_datetime_features(df)
    df = add_angle_encodings(df)
    df = add_spatial_vector_features(df)

    for feature in ml2_feature_columns:
        if feature not in df.columns:
            df[feature] = np.nan

    X = df[ml2_feature_columns].copy()

    return X, df


def load_ml2_model_artifact(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Final ML-2 model not found:\n{path}\n\n"
            "Run 09_train_ml2_plume_direction_model.py first."
        )

    artifact = joblib.load(path)

    if isinstance(artifact, dict):

        if "pipeline" not in artifact:
            raise RuntimeError(
                "ML-2 model artifact does not contain a pipeline."
            )

        return artifact

    return {
        "pipeline": artifact,
        "target_columns": TARGET_COLUMNS,
        "feature_columns": None,
    }


def run_ml2_prediction(ml2_ready_df, ml2_model_artifact, ml2_feature_config):
    pipeline = ml2_model_artifact["pipeline"]

    if ml2_model_artifact.get("feature_columns") is not None:
        ml2_feature_columns = ml2_model_artifact["feature_columns"]
    else:
        ml2_feature_columns = ml2_feature_config["selected_features"]

    X, prepared_df = prepare_ml2_prediction_features(
        ml2_ready_df,
        ml2_feature_columns
    )

    predictions = pipeline.predict(X)

    prepared_df["ml2_predicted_plume_u"] = predictions[:, 0]
    prepared_df["ml2_predicted_plume_v"] = predictions[:, 1]

    prepared_df["ml2_predicted_plume_angle"] = (
        np.degrees(
            np.arctan2(
                prepared_df["ml2_predicted_plume_v"],
                prepared_df["ml2_predicted_plume_u"]
            )
        )
        + 360
    ) % 360

    prepared_df["ml2_predicted_plume_strength"] = np.sqrt(
        prepared_df["ml2_predicted_plume_u"] ** 2
        + prepared_df["ml2_predicted_plume_v"] ** 2
    )

    return prepared_df, ml2_feature_columns


def add_plume_comparison_columns(df):
    df = df.copy()

    if "target_plume_u" in df.columns and "target_plume_v" in df.columns:

        df["simulation_label_plume_angle"] = (
            np.degrees(
                np.arctan2(
                    df["target_plume_v"],
                    df["target_plume_u"]
                )
            )
            + 360
        ) % 360

        df["simulation_label_plume_strength"] = np.sqrt(
            df["target_plume_u"] ** 2
            + df["target_plume_v"] ** 2
        )

        angle_difference = (
            df["ml2_predicted_plume_angle"]
            - df["simulation_label_plume_angle"]
        ).abs()

        df["final_angular_error"] = np.minimum(
            angle_difference,
            360 - angle_difference
        )

        df["final_vector_error"] = np.sqrt(
            (
                df["ml2_predicted_plume_u"]
                - df["target_plume_u"]
            ) ** 2
            + (
                df["ml2_predicted_plume_v"]
                - df["target_plume_v"]
            ) ** 2
        )

    return df


def get_coordinate_columns(df):
    source_lat_col = get_first_existing_column(
        df,
        [
            "source_lat",
            "industry_lat",
            "emission_lat",
        ]
    )

    source_lon_col = get_first_existing_column(
        df,
        [
            "source_lon",
            "industry_lon",
            "emission_lon",
        ]
    )

    grid_lat_col = get_first_existing_column(
        df,
        [
            "grid_lat",
            "cell_lat",
            "receiver_lat",
            "latitude",
        ]
    )

    grid_lon_col = get_first_existing_column(
        df,
        [
            "grid_lon",
            "cell_lon",
            "receiver_lon",
            "longitude",
        ]
    )

    return source_lat_col, source_lon_col, grid_lat_col, grid_lon_col


def get_pollution_strength(row):
    for column in ["ml1_predicted_pm25", "pm25", "ml1_predicted_pm10", "pm10"]:
        if column in row.index:
            value = row[column]

            if pd.notna(value):
                value = float(value)

                if value > 0:
                    return value

    return 1.0


def get_rainfall_factor(row):
    rainfall = 0

    if "rainfall" in row.index and pd.notna(row["rainfall"]):
        rainfall = float(row["rainfall"])

    rainfall_factor = 1 / (1 + rainfall)

    return rainfall_factor


def build_particle_frames_for_record(row, row_id, source_lat_col, source_lon_col):
    rng = np.random.default_rng(RANDOM_STATE + int(row_id))

    source_lat = float(row[source_lat_col])
    source_lon = float(row[source_lon_col])

    plume_u = float(row["ml2_predicted_plume_u"])
    plume_v = float(row["ml2_predicted_plume_v"])

    plume_strength = float(
        np.sqrt(plume_u ** 2 + plume_v ** 2)
    )

    pollution_strength = get_pollution_strength(row)
    rainfall_factor = get_rainfall_factor(row)

    particle_size = 3.2 + min(pollution_strength / 80, 3)
    base_opacity = 0.35 + min(pollution_strength / 300, 0.35)
    base_opacity = base_opacity * rainfall_factor
    base_opacity = max(0.12, min(base_opacity, 0.75))

    frames = []
    flat_frame_rows = []

    for frame_index in range(PLUME_FRAME_COUNT):

        if PLUME_FRAME_COUNT <= 1:
            progress = 0
        else:
            progress = frame_index / (PLUME_FRAME_COUNT - 1)

        center_lat = source_lat + plume_v * PLUME_STEP_SCALE * progress
        center_lon = source_lon + plume_u * PLUME_STEP_SCALE * progress

        spread = (
            PARTICLE_SPREAD_SCALE
            * (1 + progress * 3)
            * (1 + plume_strength * 0.05)
        )

        particles = []

        for particle_index in range(PARTICLES_PER_FRAME):

            particle_lat = center_lat + rng.normal(0, spread)
            particle_lon = center_lon + rng.normal(0, spread)

            opacity = base_opacity * (1 - progress * 0.35)
            opacity = max(0.08, min(opacity, 0.75))

            particle = {
                "lat": round(float(particle_lat), 6),
                "lon": round(float(particle_lon), 6),
                "size": round(float(particle_size), 3),
                "opacity": round(float(opacity), 3),
            }

            particles.append(particle)

            flat_frame_rows.append(
                {
                    "record_id": int(row_id),
                    "frame_index": int(frame_index),
                    "particle_index": int(particle_index),
                    "lat": particle["lat"],
                    "lon": particle["lon"],
                    "size": particle["size"],
                    "opacity": particle["opacity"],
                    "center_lat": round(float(center_lat), 6),
                    "center_lon": round(float(center_lon), 6),
                }
            )

        frames.append(
            {
                "frame_index": int(frame_index),
                "center_lat": round(float(center_lat), 6),
                "center_lon": round(float(center_lon), 6),
                "particles": particles,
            }
        )

    return frames, flat_frame_rows


def build_html_simulation_data(final_df):
    source_lat_col, source_lon_col, grid_lat_col, grid_lon_col = get_coordinate_columns(final_df)

    simulation_df = final_df.copy()

    if source_lat_col is None or source_lon_col is None:
        print("Source coordinates were missing.")
        print("Using prototype fallback source coordinates for Byrnihat.")

        simulation_df["source_lat"] = DEFAULT_SOURCE_LAT
        simulation_df["source_lon"] = DEFAULT_SOURCE_LON
        simulation_df["source_name"] = DEFAULT_SOURCE_NAME

        source_lat_col = "source_lat"
        source_lon_col = "source_lon"

    simulation_df[source_lat_col] = pd.to_numeric(
        simulation_df[source_lat_col],
        errors="coerce"
    )

    simulation_df[source_lon_col] = pd.to_numeric(
        simulation_df[source_lon_col],
        errors="coerce"
    )

    simulation_df[source_lat_col] = simulation_df[source_lat_col].fillna(
        DEFAULT_SOURCE_LAT
    )

    simulation_df[source_lon_col] = simulation_df[source_lon_col].fillna(
        DEFAULT_SOURCE_LON
    )

    if "source_name" not in simulation_df.columns:
        simulation_df["source_name"] = DEFAULT_SOURCE_NAME

    simulation_df["source_name"] = simulation_df["source_name"].fillna(
        DEFAULT_SOURCE_NAME
    )

    simulation_df = simulation_df.dropna(
        subset=[
            source_lat_col,
            source_lon_col,
            "ml2_predicted_plume_u",
            "ml2_predicted_plume_v",
        ]
    ).reset_index(drop=True)

    if simulation_df.empty:
        raise RuntimeError(
            "No rows available for ML-predicted plume animation."
        )

    if len(simulation_df) > MAX_RECORDS_FOR_HTML:
        simulation_df = simulation_df.tail(MAX_RECORDS_FOR_HTML).reset_index(drop=True)

    records = []
    all_flat_frame_rows = []

    for row_id, row in simulation_df.iterrows():

        frames, flat_frame_rows = build_particle_frames_for_record(
            row,
            row_id,
            source_lat_col,
            source_lon_col
        )

        all_flat_frame_rows.extend(flat_frame_rows)

        record = {
            "record_id": int(row_id),
            "datetime": safe_value(row["datetime"]) if "datetime" in row.index else None,
            "station_name": safe_value(row["station_name"]) if "station_name" in row.index else None,
            "source_name": safe_value(row["source_name"]) if "source_name" in row.index else None,
            "industry_name": safe_value(row["industry_name"]) if "industry_name" in row.index else None,
            "source": {
                "lat": safe_value(row[source_lat_col]),
                "lon": safe_value(row[source_lon_col]),
                "lat_column": source_lat_col,
                "lon_column": source_lon_col,
            },
            "grid": {
                "lat": safe_value(row[grid_lat_col]) if grid_lat_col is not None else None,
                "lon": safe_value(row[grid_lon_col]) if grid_lon_col is not None else None,
                "lat_column": grid_lat_col,
                "lon_column": grid_lon_col,
            },
            "recorded": {},
            "ml1_prediction": {},
            "ml2_prediction": {
                "plume_u": safe_value(row["ml2_predicted_plume_u"]),
                "plume_v": safe_value(row["ml2_predicted_plume_v"]),
                "plume_angle": safe_value(row["ml2_predicted_plume_angle"]),
                "plume_strength": safe_value(row["ml2_predicted_plume_strength"]),
            },
            "simulation_label": {},
            "frames": frames,
        }

        for column in POLLUTION_COLUMNS + WEATHER_COLUMNS:
            if column in row.index:
                record["recorded"][column] = safe_value(row[column])

        for column in simulation_df.columns:
            if column.startswith("ml1_predicted_"):
                record["ml1_prediction"][column] = safe_value(row[column])

        if "target_plume_u" in row.index and "target_plume_v" in row.index:
            record["simulation_label"] = {
                "target_plume_u": safe_value(row["target_plume_u"]),
                "target_plume_v": safe_value(row["target_plume_v"]),
                "plume_angle": safe_value(row["simulation_label_plume_angle"]) if "simulation_label_plume_angle" in row.index else None,
                "plume_strength": safe_value(row["simulation_label_plume_strength"]) if "simulation_label_plume_strength" in row.index else None,
                "angular_error": safe_value(row["final_angular_error"]) if "final_angular_error" in row.index else None,
                "vector_error": safe_value(row["final_vector_error"]) if "final_vector_error" in row.index else None,
            }

        records.append(record)

    flat_frames_df = pd.DataFrame(all_flat_frame_rows)

    simulation_data = {
        "metadata": {
            "title": "Byrnihat Wind + DEM + ML Pollution Plume Simulation",
            "description": "ML-2 predicted plume movement data for the existing Leaflet/Folium HTML simulation.",
            "movement_source": "ML-2 predicted plume vector",
            "source_coordinate_mode": "prototype_fallback_if_missing",
            "prototype_fallback_source_lat": DEFAULT_SOURCE_LAT,
            "prototype_fallback_source_lon": DEFAULT_SOURCE_LON,
            "frame_count_per_record": PLUME_FRAME_COUNT,
            "particles_per_frame": PARTICLES_PER_FRAME,
            "plume_step_scale": PLUME_STEP_SCALE,
            "particle_spread_scale": PARTICLE_SPREAD_SCALE,
            "max_records_for_html": MAX_RECORDS_FOR_HTML,
            "source_lat_column": source_lat_col,
            "source_lon_column": source_lon_col,
            "grid_lat_column": grid_lat_col,
            "grid_lon_column": grid_lon_col,
            "record_count": len(records),
        },
        "records": records,
    }

    return simulation_data, flat_frames_df


def add_series_if_available(graph_data, group_name, series_name, df, actual_col, predicted_col):
    if actual_col not in df.columns:
        return

    if predicted_col not in df.columns:
        return

    rows = []

    for _, row in df.iterrows():
        rows.append(
            {
                "datetime": safe_value(row["datetime"]) if "datetime" in row.index else None,
                "recorded": safe_value(row[actual_col]),
                "predicted": safe_value(row[predicted_col]),
            }
        )

    graph_data[group_name][series_name] = rows


def add_ml2_series_if_available(graph_data, series_name, df, label_col, prediction_col):
    if label_col not in df.columns:
        return

    if prediction_col not in df.columns:
        return

    rows = []

    for _, row in df.iterrows():
        rows.append(
            {
                "datetime": safe_value(row["datetime"]) if "datetime" in row.index else None,
                "simulation_label": safe_value(row[label_col]),
                "ml2_predicted": safe_value(row[prediction_col]),
            }
        )

    graph_data["ml2_simulation_label_vs_prediction"][series_name] = rows


def build_graph_data(final_df):
    graph_df = final_df.copy()

    if len(graph_df) > MAX_RECORDS_FOR_HTML:
        graph_df = graph_df.tail(MAX_RECORDS_FOR_HTML).reset_index(drop=True)

    graph_data = {
        "metadata": {
            "title": "Byrnihat final dashboard graph data",
            "description": "Recorded vs predicted ML data for dashboard charts.",
            "record_count": len(graph_df),
        },
        "ml1_recorded_vs_predicted": {},
        "ml2_simulation_label_vs_prediction": {},
        "error_timeseries": {},
    }

    add_series_if_available(
        graph_data,
        "ml1_recorded_vs_predicted",
        "pm25",
        graph_df,
        "pm25",
        "ml1_predicted_pm25"
    )

    add_series_if_available(
        graph_data,
        "ml1_recorded_vs_predicted",
        "pm10",
        graph_df,
        "pm10",
        "ml1_predicted_pm10"
    )

    add_series_if_available(
        graph_data,
        "ml1_recorded_vs_predicted",
        "wind_u",
        graph_df,
        "wind_u",
        "ml1_predicted_wind_u"
    )

    add_series_if_available(
        graph_data,
        "ml1_recorded_vs_predicted",
        "wind_v",
        graph_df,
        "wind_v",
        "ml1_predicted_wind_v"
    )

    add_ml2_series_if_available(
        graph_data,
        "plume_u",
        graph_df,
        "target_plume_u",
        "ml2_predicted_plume_u"
    )

    add_ml2_series_if_available(
        graph_data,
        "plume_v",
        graph_df,
        "target_plume_v",
        "ml2_predicted_plume_v"
    )

    add_ml2_series_if_available(
        graph_data,
        "plume_angle",
        graph_df,
        "simulation_label_plume_angle",
        "ml2_predicted_plume_angle"
    )

    error_rows = []

    if "final_angular_error" in graph_df.columns or "final_vector_error" in graph_df.columns:

        for _, row in graph_df.iterrows():
            error_rows.append(
                {
                    "datetime": safe_value(row["datetime"]) if "datetime" in row.index else None,
                    "angular_error": safe_value(row["final_angular_error"]) if "final_angular_error" in row.index else None,
                    "vector_error": safe_value(row["final_vector_error"]) if "final_vector_error" in row.index else None,
                }
            )

    graph_data["error_timeseries"]["plume_prediction_errors"] = error_rows

    return graph_data


def select_final_output_columns(df):
    df = df.copy()

    important_columns = [
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

        "target_plume_u",
        "target_plume_v",
        "simulation_label_plume_angle",
        "simulation_label_plume_strength",

        "ml2_predicted_plume_u",
        "ml2_predicted_plume_v",
        "ml2_predicted_plume_angle",
        "ml2_predicted_plume_strength",

        "final_angular_error",
        "final_vector_error",
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

    return df[selected_columns + remaining_columns]


def main():
    print("Project root:")
    print(PROJECT_ROOT)

    print("\nLoading ML-1 base dataset:")
    print(ML1_BASE_DATA_FILE)

    ml1_base_df = load_csv_file(
        ML1_BASE_DATA_FILE,
        "ML-1 base dataset"
    )

    ml1_base_df = prepare_datetime(ml1_base_df)
    ml1_base_df = convert_numeric_like_columns(ml1_base_df)

    print("ML-1 base rows:")
    print(len(ml1_base_df))

    print("\nLoading ML-1 feature config:")
    print(ML1_FEATURE_CONFIG_FILE)

    ml1_feature_config = load_json_file(
        ML1_FEATURE_CONFIG_FILE,
        "ML-1 feature config"
    )

    print("\nCreating ML-1 prediction columns...")
    ml1_prediction_df, ml1_prediction_report = create_ml1_prediction_features(
        ml1_base_df,
        ml1_feature_config
    )

    ml1_prediction_df = add_ml1_predicted_wind_features(ml1_prediction_df)

    print("ML-1 prediction report:")
    print(ml1_prediction_report)

    print("\nLoading ML-2 base dataset:")
    print(ML2_BASE_DATA_FILE)

    ml2_base_df = load_csv_file(
        ML2_BASE_DATA_FILE,
        "ML-2 base dataset"
    )

    ml2_base_df = prepare_datetime(ml2_base_df)
    ml2_base_df = convert_numeric_like_columns(ml2_base_df)

    print("ML-2 base rows:")
    print(len(ml2_base_df))

    print("\nMerging ML-1 predictions with ML-2 base data...")
    ml2_ready_df, merge_keys, ml1_prediction_columns = merge_ml1_predictions_with_ml2_base(
        ml2_base_df,
        ml1_prediction_df
    )

    print("Merge keys used:")
    print(merge_keys)

    print("ML-1 prediction columns merged:")
    print(ml1_prediction_columns)

    print("\nLoading ML-2 feature config:")
    print(ML2_FEATURE_CONFIG_FILE)

    ml2_feature_config = load_json_file(
        ML2_FEATURE_CONFIG_FILE,
        "ML-2 feature config"
    )

    print("\nLoading ML-2 model:")
    print(ML2_MODEL_FILE)

    ml2_model_artifact = load_ml2_model_artifact(ML2_MODEL_FILE)

    print("\nRunning final ML-2 plume prediction...")
    final_df, ml2_feature_columns = run_ml2_prediction(
        ml2_ready_df,
        ml2_model_artifact,
        ml2_feature_config
    )

    final_df = add_plume_comparison_columns(final_df)

    if len(final_df) < MIN_FINAL_ROWS:
        raise RuntimeError("Final prediction dataframe is empty.")

    final_df = select_final_output_columns(final_df)

    print("\nFinal prediction rows:")
    print(len(final_df))

    print("\nSaving final prediction output...")
    final_df.to_csv(FINAL_PREDICTION_FILE, index=False)

    print(FINAL_PREDICTION_FILE)

    print("\nBuilding graph data...")
    graph_data = build_graph_data(final_df)
    save_json(graph_data, FINAL_GRAPH_DATA_FILE)

    print(FINAL_GRAPH_DATA_FILE)

    print("\nBuilding ML plume data for existing HTML simulation...")
    simulation_data, plume_frames_df = build_html_simulation_data(final_df)

    save_json(simulation_data, ML_PLUME_SIMULATION_JSON_FILE)
    save_js_variable(
        "ML_PLUME_SIMULATION_DATA",
        simulation_data,
        ML_PLUME_SIMULATION_JS_FILE
    )

    plume_frames_df.to_csv(ML_PLUME_FRAMES_CSV_FILE, index=False)

    print(ML_PLUME_SIMULATION_JSON_FILE)
    print(ML_PLUME_SIMULATION_JS_FILE)
    print(ML_PLUME_FRAMES_CSV_FILE)

    pipeline_report = {
        "script_name": "10_run_final_prediction_pipeline.py",
        "purpose": "Run final ML-1 plus ML-2 prediction pipeline and prepare dashboard-ready outputs.",
        "ml1_base_data_file": str(ML1_BASE_DATA_FILE),
        "ml2_base_data_file": str(ML2_BASE_DATA_FILE),
        "ml1_feature_config_file": str(ML1_FEATURE_CONFIG_FILE),
        "ml2_feature_config_file": str(ML2_FEATURE_CONFIG_FILE),
        "ml2_model_file": str(ML2_MODEL_FILE),
        "final_prediction_file": str(FINAL_PREDICTION_FILE),
        "final_graph_data_file": str(FINAL_GRAPH_DATA_FILE),
        "ml_plume_simulation_json_file": str(ML_PLUME_SIMULATION_JSON_FILE),
        "ml_plume_simulation_js_file": str(ML_PLUME_SIMULATION_JS_FILE),
        "ml_plume_frames_csv_file": str(ML_PLUME_FRAMES_CSV_FILE),
        "merge_keys": merge_keys,
        "ml1_prediction_columns": ml1_prediction_columns,
        "ml2_feature_columns": ml2_feature_columns,
        "final_rows": len(final_df),
        "final_columns": final_df.columns.tolist(),
        "ml1_prediction_report": ml1_prediction_report,
        "prototype_source_fallback_used_if_source_coordinates_missing": True,
        "default_source_lat": DEFAULT_SOURCE_LAT,
        "default_source_lon": DEFAULT_SOURCE_LON,
    }

    save_json(pipeline_report, PIPELINE_REPORT_FILE)

    print("\nSaved pipeline report:")
    print(PIPELINE_REPORT_FILE)

    print("\nFinal prediction pipeline completed successfully.")


if __name__ == "__main__":
    main()