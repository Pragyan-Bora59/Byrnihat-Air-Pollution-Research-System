"""
backend/analysis_engine.py

Wraps analysis/observed_data_analysis_v1.py logic into clean, importable functions.
The original script is NOT modified. Functions return DataFrames/dicts — never HTML.
"""

import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Internal helpers (mirrors original script logic)
# ---------------------------------------------------------------------------

def _normalize_column_name(column):
    column = str(column).strip().lower()
    for char, repl in [(" ", "_"), (".", "_"), ("-", "_"), ("/", "_"),
                       ("(", ""), (")", "")]:
        column = column.replace(char, repl)
    return column


def _normalize_column_names(df):
    df = df.copy()
    df.columns = [_normalize_column_name(c) for c in df.columns]
    return df


def read_city_file(file_path, city_name):
    """
    Read one city AQICN CSV file and return a cleaned DataFrame.
    Mirrors read_city_file() from analysis/observed_data_analysis_v1.py.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return pd.DataFrame()

    data = pd.read_csv(file_path)

    # Drop unnamed index columns
    unwanted = [c for c in data.columns
                if c.startswith("Unnamed") or c == "0"]
    data = data.drop(columns=unwanted, errors="ignore")

    if "city" not in data.columns:
        data["city"] = city_name

    if "api_time" in data.columns:
        data["api_time"] = pd.to_datetime(data["api_time"], errors="coerce")

    if "recorded_at_local_time" in data.columns:
        data["recorded_at_local_time"] = pd.to_datetime(
            data.get("api_time"), errors="coerce"
        )

    numeric_columns = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "api_time" in data.columns:
        data = data.dropna(subset=["api_time"])
        data = data.drop_duplicates(subset=["city", "api_time"], keep="last")

    return data


def load_all_city_data():
    """
    Load and combine AQICN data for Byrnihat, Guwahati, and Shillong.
    Returns a combined DataFrame sorted by city and time.
    """
    city_files = {
        "Byrnihat": DATA_DIR / "byrnihat_aqicn_data.csv",
        "Guwahati": DATA_DIR / "guwahati_aqicn_data.csv",
        "Shillong": DATA_DIR / "shillong_aqicn_data.csv",
    }

    frames = []
    for city_name, fpath in city_files.items():
        df = read_city_file(fpath, city_name)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Sort by time
    time_col = "api_time" if "api_time" in combined.columns else "datetime"
    if time_col in combined.columns:
        combined = combined.sort_values(["city", time_col])

    return combined


def compute_city_summary(df):
    """
    Compute per-city summary statistics (count, mean, max, min) for pollutants.
    Returns a DataFrame — mirrors the summary_table from the original script.
    """
    pollutant_cols = [c for c in ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co"]
                      if c in df.columns]

    if not pollutant_cols or "city" not in df.columns:
        return pd.DataFrame()

    summary = df.groupby("city")[pollutant_cols].agg(["count", "mean", "max", "min"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary = summary.round(2)

    return summary


def compute_missing_report(df):
    """
    Compute per-city missing value counts.
    Returns a DataFrame — mirrors the missing_report from the original script.
    """
    if "city" not in df.columns:
        return df.isna().sum().to_frame("missing_count")

    report = df.groupby("city").apply(lambda g: g.isna().sum())
    return report


def get_latest_values(df):
    """
    Get the latest observation row for each city.
    Returns a DataFrame with one row per city.
    Mirrors the latest_values logic from the original script.
    """
    time_col = "api_time" if "api_time" in df.columns else "datetime"
    if time_col not in df.columns or "city" not in df.columns:
        return df.tail(1)

    latest_rows = []
    for city in df["city"].unique():
        city_data = (df[df["city"] == city]
                     .sort_values(time_col)
                     .dropna(subset=[time_col]))
        if len(city_data) > 0:
            latest_rows.append(city_data.iloc[-1])

    if not latest_rows:
        return pd.DataFrame()

    return pd.DataFrame(latest_rows).reset_index(drop=True)


def get_aqi_category(aqi_value):
    """
    Return the AQI category label and a CSS-compatible colour for a given AQI value.
    Based on India CPCB AQI categories.
    """
    if pd.isna(aqi_value) or aqi_value is None:
        return "Unknown", "#64748b"

    aqi = float(aqi_value)

    if aqi <= 50:
        return "Good", "#22c55e"
    elif aqi <= 100:
        return "Satisfactory", "#84cc16"
    elif aqi <= 200:
        return "Moderate", "#f59e0b"
    elif aqi <= 300:
        return "Poor", "#f97316"
    elif aqi <= 400:
        return "Very Poor", "#ef4444"
    else:
        return "Severe", "#7c3aed"


def get_pollution_trend(df, pollutant="aqi", city=None, last_n=7):
    """
    Get the last N days of data for a pollutant (for sparkline charts).
    Returns a small DataFrame suitable for a mini chart.
    """
    time_col = "api_time" if "api_time" in df.columns else "datetime"

    if pollutant not in df.columns or time_col not in df.columns:
        return pd.DataFrame()

    data = df.copy()

    if city and "city" in data.columns:
        data = data[data["city"] == city]

    data = data.sort_values(time_col).dropna(subset=[time_col, pollutant])

    if data.empty:
        return pd.DataFrame()

    # Resample to daily mean for last N days
    data = data.set_index(time_col)
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data[[pollutant]].resample("D").mean().dropna()
    data = data.tail(last_n).reset_index()

    return data
