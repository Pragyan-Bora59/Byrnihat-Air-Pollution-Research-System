from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# 1. Paths and configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

OUTPUT_FILE = PROCESSED_DIR / "ml1_environment_pollution_dataset.csv"
MISSING_REPORT_FILE = REPORTS_DIR / "ml1_dataset_missing_report.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_TOLERANCE_HOURS = 6

POLLUTANT_COLUMNS = [
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
]


# =============================================================================
# 2. Basic helper functions
# =============================================================================

def clean_column_names(df):
    df = df.copy()

    df.columns = (
        pd.Index(df.columns)
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


def drop_noise_columns(df):
    df = df.copy()

    columns_to_drop = []

    for column in df.columns:
        column_string = str(column).strip().lower()

        if column_string == "":
            columns_to_drop.append(column)
        elif column_string == "0":
            columns_to_drop.append(column)
        elif column_string.startswith("unnamed"):
            columns_to_drop.append(column)

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    return df


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


def find_prototype_dir():
    expected_dir = (
        DATA_DIR
        / "prototype"
        / "byrnihat_missing_source_proxy_datasets"
    )

    if expected_dir.exists():
        return expected_dir

    prototype_root = DATA_DIR / "prototype"

    if prototype_root.exists():
        manifest_files = list(prototype_root.rglob("dataset_manifest.csv"))

        if manifest_files:
            return manifest_files[0].parent

    manifest_files = list(DATA_DIR.rglob("dataset_manifest.csv"))

    if manifest_files:
        return manifest_files[0].parent

    raise FileNotFoundError(
        "Prototype dataset folder not found.\n\n"
        "Expected location:\n"
        f"{expected_dir}\n\n"
        "Make sure this folder exists:\n"
        "data/prototype/byrnihat_missing_source_proxy_datasets"
    )


def find_first_file(folder, patterns):
    for pattern in patterns:
        files = sorted(folder.rglob(pattern))

        if files:
            return files[0]

    return None


def add_wind_vectors(df):
    df = df.copy()

    if "wind_speed" not in df.columns:
        df["wind_speed"] = np.nan

    if "wind_direction" not in df.columns:
        df["wind_direction"] = np.nan

    theta = np.deg2rad(df["wind_direction"])

    df["wind_u"] = -df["wind_speed"] * np.sin(theta)
    df["wind_v"] = -df["wind_speed"] * np.cos(theta)

    return df


def reconstruct_wind_from_vectors(df):
    df = df.copy()

    if "wind_u" in df.columns and "wind_v" in df.columns:
        df["wind_speed"] = np.sqrt(
            df["wind_u"] ** 2
            + df["wind_v"] ** 2
        )

        df["wind_to_direction"] = (
            np.degrees(
                np.arctan2(
                    df["wind_u"],
                    df["wind_v"]
                )
            )
            + 360
        ) % 360

        df["wind_direction"] = (df["wind_to_direction"] + 180) % 360

    return df


def add_time_features(df):
    df = df.copy()

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


def convert_numeric_columns(df):
    df = df.copy()

    numeric_columns = [
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
        "aqicn_temperature",
        "aqicn_humidity",
        "aqicn_pressure",
        "aqicn_wind_indicator",
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
        "hour",
        "day",
        "month",
        "day_of_week",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def order_final_columns(df):
    df = df.copy()

    preferred_columns = [
        "datetime",
        "target_city",
        "station_name",
        "aqicn_station_name",
        "station_code",
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

        "aqicn_temperature",
        "aqicn_humidity",
        "aqicn_pressure",
        "aqicn_wind_indicator",

        "temperature",
        "humidity",
        "pressure",
        "rainfall",
        "wind_speed",
        "wind_direction",
        "wind_to_direction",
        "wind_u",
        "wind_v",

        "hour",
        "day",
        "month",
        "day_of_week",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",

        "weather_station_name",
        "weather_datetime",
        "weather_gap_hours",
        "weather_merge_mode",
        "is_prototype_weather_match",

        "data_status",
        "intended_use",
        "source_note",
    ]

    existing_preferred_columns = []

    for column in preferred_columns:
        if column in df.columns:
            existing_preferred_columns.append(column)

    remaining_columns = []

    for column in df.columns:
        if column not in existing_preferred_columns:
            remaining_columns.append(column)

    df = df[existing_preferred_columns + remaining_columns]

    return df


# =============================================================================
# 3. Direct ML-ready prototype loader
# =============================================================================

def load_ml_ready_prototype_file(prototype_dir):
    print("\n" + "=" * 80)
    print("CHECKING FOR ML-READY PROTOTYPE DATASET")
    print("=" * 80)

    ml_ready_file = find_first_file(
        prototype_dir,
        [
            "prototype_ml1_pollution_meteorology_dataset.csv",
        ]
    )

    if ml_ready_file is None:
        print("No ML-ready prototype ML-1 file found.")
        return None

    print("Using ML-ready prototype file:")
    print(ml_ready_file)

    df = pd.read_csv(ml_ready_file)
    df = clean_column_names(df)
    df = drop_noise_columns(df)

    if "datetime" not in df.columns and "api_time" in df.columns:
        df["datetime"] = df["api_time"]

    if "datetime" not in df.columns:
        raise RuntimeError(
            "ML-ready prototype file must contain datetime or api_time."
        )

    df["datetime"] = parse_datetime_naive(df["datetime"])
    df = df.dropna(subset=["datetime"])

    if "target_city" in df.columns:
        if "station_name" in df.columns:
            df["aqicn_station_name"] = df["station_name"]

        df["station_name"] = df["target_city"]

    if "station_latitude" in df.columns:
        df["latitude"] = df["station_latitude"]

    if "station_longitude" in df.columns:
        df["longitude"] = df["station_longitude"]

    met_column_map = {
        "met_temperature": "temperature",
        "met_humidity": "humidity",
        "met_pressure": "pressure",
        "met_rainfall": "rainfall",
        "met_wind_speed": "wind_speed",
        "met_wind_direction": "wind_direction",
        "met_wind_to_direction": "wind_to_direction",
        "met_wind_u": "wind_u",
        "met_wind_v": "wind_v",
    }

    for old_column, new_column in met_column_map.items():
        if old_column in df.columns:
            df[new_column] = df[old_column]

    if "weather_station_name" not in df.columns:
        df["weather_station_name"] = "prototype_weather"

    df["weather_datetime"] = df["datetime"]
    df["weather_gap_hours"] = 0
    df["weather_merge_mode"] = "prototype_ml_ready_file"
    df["is_prototype_weather_match"] = True

    df = add_time_features(df)

    if "wind_u" not in df.columns or "wind_v" not in df.columns:
        df = add_wind_vectors(df)

    df = reconstruct_wind_from_vectors(df)
    df = convert_numeric_columns(df)
    df = order_final_columns(df)

    df = df.drop_duplicates()
    df = df.sort_values(["station_name", "datetime"]).reset_index(drop=True)

    return df


# =============================================================================
# 4. Fallback: load separate AQICN prototype files
# =============================================================================

def load_aqicn_prototype_files(prototype_dir):
    print("\n" + "=" * 80)
    print("LOADING PROTOTYPE AQICN OBSERVATION FILES")
    print("=" * 80)

    aqicn_files = sorted(
        prototype_dir.rglob("prototype_*_aqicn_observations.csv")
    )

    if not aqicn_files:
        all_city_file = find_first_file(
            prototype_dir,
            [
                "prototype_all_three_city_pollution_observations.csv",
            ]
        )

        if all_city_file is not None:
            aqicn_files = [all_city_file]

    print("Prototype directory:")
    print(prototype_dir)

    print("\nAQICN prototype files found:")
    for file_path in aqicn_files:
        print("-", file_path)

    if not aqicn_files:
        raise FileNotFoundError(
            "No prototype AQICN files found.\n\n"
            "Expected files like:\n"
            "prototype_byrnihat_aqicn_observations.csv\n"
            "prototype_guwahati_aqicn_observations.csv\n"
            "prototype_shillong_aqicn_observations.csv\n"
        )

    aqicn_parts = []

    for file_path in aqicn_files:
        df = pd.read_csv(file_path)

        df = clean_column_names(df)
        df = drop_noise_columns(df)

        if "api_time" in df.columns and "datetime" not in df.columns:
            df["datetime"] = df["api_time"]

        if "recorded_at_local_time" in df.columns and "datetime" not in df.columns:
            df["datetime"] = df["recorded_at_local_time"]

        if "station_latitude" in df.columns:
            df["latitude"] = df["station_latitude"]

        if "station_longitude" in df.columns:
            df["longitude"] = df["station_longitude"]

        if "target_city" in df.columns:
            if "station_name" in df.columns:
                df["aqicn_station_name"] = df["station_name"]

            df["station_name"] = df["target_city"]

        else:
            file_name = file_path.name.lower()

            if "byrnihat" in file_name:
                df["station_name"] = "Byrnihat"
            elif "guwahati" in file_name:
                df["station_name"] = "Guwahati"
            elif "shillong" in file_name:
                df["station_name"] = "Shillong"
            else:
                df["station_name"] = "Unknown"

            df["target_city"] = df["station_name"]

        aqicn_weather_map = {
            "temperature": "aqicn_temperature",
            "humidity": "aqicn_humidity",
            "pressure": "aqicn_pressure",
            "wind": "aqicn_wind_indicator",
        }

        for old_column, new_column in aqicn_weather_map.items():
            if old_column in df.columns:
                df[new_column] = df[old_column]

        aqicn_parts.append(df)

    aq = pd.concat(aqicn_parts, ignore_index=True)
    aq = aq.dropna(how="all")

    if "datetime" not in aq.columns:
        raise RuntimeError("AQICN prototype data must contain datetime or api_time.")

    aq["datetime"] = parse_datetime_naive(aq["datetime"])
    aq = aq.dropna(subset=["datetime"])

    aq["station_name"] = aq["station_name"].astype(str).str.strip()

    aq = convert_numeric_columns(aq)
    aq = aq.sort_values(["station_name", "datetime"]).reset_index(drop=True)

    print("\nAQICN prototype shape:")
    print(aq.shape)

    print("\nAQICN prototype columns:")
    print(aq.columns.tolist())

    return aq


# =============================================================================
# 5. Fallback: load separate meteorology prototype file
# =============================================================================

def load_weather_prototype_file(prototype_dir):
    print("\n" + "=" * 80)
    print("LOADING PROTOTYPE METEOROLOGY FILE")
    print("=" * 80)

    weather_file = find_first_file(
        prototype_dir,
        [
            "prototype_nasa_power_hourly_tracker_byrnihat_region.csv",
            "prototype_byrnihat_region_meteorological_observations.csv",
        ]
    )

    if weather_file is None:
        raise FileNotFoundError(
            "No prototype meteorology file found.\n\n"
            "Expected one of:\n"
            "prototype_nasa_power_hourly_tracker_byrnihat_region.csv\n"
            "prototype_byrnihat_region_meteorological_observations.csv"
        )

    print("Using weather prototype file:")
    print(weather_file)

    weather = pd.read_csv(weather_file)
    weather = clean_column_names(weather)
    weather = drop_noise_columns(weather)

    datetime_column = None

    datetime_candidates = [
        "datetime_ist",
        "datetime_local",
        "datetime",
        "date_time",
        "timestamp",
        "time",
        "datetime_utc",
    ]

    for candidate in datetime_candidates:
        if candidate in weather.columns:
            datetime_column = candidate
            break

    if datetime_column is None:
        raise RuntimeError(
            "Prototype meteorology file must contain a datetime column."
        )

    weather["datetime"] = parse_datetime_naive(weather[datetime_column])
    weather = weather.dropna(subset=["datetime"])

    if "station_name" in weather.columns:
        weather["weather_station_name"] = weather["station_name"].astype(str).str.strip()
    else:
        weather["weather_station_name"] = "prototype_weather"

    for column in [
        "temperature",
        "humidity",
        "pressure",
        "rainfall",
        "wind_speed",
        "wind_direction",
        "wind_to_direction",
        "wind_u",
        "wind_v",
    ]:
        if column in weather.columns:
            weather[column] = pd.to_numeric(weather[column], errors="coerce")
        else:
            weather[column] = np.nan

    if weather["wind_u"].isna().all() or weather["wind_v"].isna().all():
        weather = add_wind_vectors(weather)

    weather = reconstruct_wind_from_vectors(weather)

    weather["weather_datetime"] = weather["datetime"]

    weather_numeric_columns = [
        "temperature",
        "humidity",
        "pressure",
        "rainfall",
        "wind_u",
        "wind_v",
    ]

    existing_numeric_columns = []

    for column in weather_numeric_columns:
        if column in weather.columns:
            existing_numeric_columns.append(column)

    weather_grouped = (
        weather
        .groupby("datetime", as_index=False)[existing_numeric_columns]
        .mean()
    )

    weather_grouped = reconstruct_wind_from_vectors(weather_grouped)

    weather_grouped["weather_station_name"] = "prototype_regional_average"
    weather_grouped["weather_datetime"] = weather_grouped["datetime"]

    weather_grouped = weather_grouped.sort_values("datetime").reset_index(drop=True)

    print("\nWeather prototype shape:")
    print(weather_grouped.shape)

    print("\nWeather prototype columns:")
    print(weather_grouped.columns.tolist())

    return weather_grouped


# =============================================================================
# 6. Fallback merge: AQICN + weather
# =============================================================================

def merge_aqicn_and_weather(aq, weather):
    print("\n" + "=" * 80)
    print("MERGING PROTOTYPE AQICN + PROTOTYPE WEATHER")
    print("=" * 80)

    aq = aq.copy()
    weather = weather.copy()

    aq = aq.sort_values("datetime").reset_index(drop=True)
    weather = weather.sort_values("datetime").reset_index(drop=True)

    ml1 = pd.merge_asof(
        aq,
        weather,
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta(hours=WEATHER_TOLERANCE_HOURS)
    )

    if ml1.empty:
        raise RuntimeError("AQICN + weather merge created an empty ML-1 dataset.")

    ml1["weather_merge_mode"] = "prototype_nearest_datetime"
    ml1["is_prototype_weather_match"] = True

    if "weather_datetime" in ml1.columns:
        ml1["weather_gap_hours"] = (
            (ml1["datetime"] - ml1["weather_datetime"])
            .abs()
            .dt.total_seconds()
            / 3600
        )
    else:
        ml1["weather_datetime"] = ml1["datetime"]
        ml1["weather_gap_hours"] = 0

    ml1 = add_time_features(ml1)

    if "wind_u" not in ml1.columns or "wind_v" not in ml1.columns:
        ml1 = add_wind_vectors(ml1)

    ml1 = reconstruct_wind_from_vectors(ml1)
    ml1 = convert_numeric_columns(ml1)
    ml1 = order_final_columns(ml1)

    ml1 = ml1.drop_duplicates()
    ml1 = ml1.sort_values(["station_name", "datetime"]).reset_index(drop=True)

    return ml1


# =============================================================================
# 7. Main ML-1 dataset creation
# =============================================================================

def create_ml1_dataset():
    print("=" * 80)
    print("CREATING ML-1 DATASET USING PROTOTYPE DATA")
    print("=" * 80)

    prototype_dir = find_prototype_dir()

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nPrototype directory being used:")
    print(prototype_dir)

    print("\nPrototype directory exists:")
    print(prototype_dir.exists())

    ml1 = load_ml_ready_prototype_file(prototype_dir)

    if ml1 is not None:
        print("\nML-1 dataset created directly from ML-ready prototype file.")
        return ml1

    print("\nFalling back to separate AQICN + meteorology prototype merge.")

    aq = load_aqicn_prototype_files(prototype_dir)
    weather = load_weather_prototype_file(prototype_dir)

    ml1 = merge_aqicn_and_weather(aq, weather)

    return ml1


# =============================================================================
# 8. Save outputs
# =============================================================================

def save_outputs(ml1):
    ml1.to_csv(OUTPUT_FILE, index=False)

    missing_report = pd.DataFrame(
        {
            "column": ml1.columns,
            "missing_count": ml1.isna().sum().values,
            "missing_percent": (ml1.isna().mean().values * 100).round(2),
        }
    )

    missing_report.to_csv(MISSING_REPORT_FILE, index=False)

    print("\n" + "=" * 80)
    print("FINAL ML-1 DATASET CREATED")
    print("=" * 80)

    print("\nSaved ML-1 dataset to:")
    print(OUTPUT_FILE)

    print("\nSaved missing report to:")
    print(MISSING_REPORT_FILE)

    print("\nFinal ML-1 shape:")
    print(ml1.shape)

    print("\nFinal ML-1 columns:")
    print(ml1.columns.tolist())

    print("\nAvailable pollutant columns:")
    print([column for column in POLLUTANT_COLUMNS if column in ml1.columns])

    print("\nWeather merge mode counts:")
    if "weather_merge_mode" in ml1.columns:
        print(ml1["weather_merge_mode"].value_counts(dropna=False))

    print("\nPrototype weather match counts:")
    if "is_prototype_weather_match" in ml1.columns:
        print(ml1["is_prototype_weather_match"].value_counts(dropna=False))

    print("\nTop missing values:")
    print(
        missing_report
        .sort_values("missing_percent", ascending=False)
        .head(20)
    )

    print("\nFirst 10 rows:")
    print(ml1.head(10))

    print("\nML-1 prototype dataset preparation completed.")


# =============================================================================
# 9. Run
# =============================================================================

def main():
    ml1 = create_ml1_dataset()
    save_outputs(ml1)


if __name__ == "__main__":
    main()