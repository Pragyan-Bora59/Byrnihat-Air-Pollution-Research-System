"""
backend/data_loader.py

Central data loading module for the Byrnihat Streamlit app.
All functions are decorated with @st.cache_data so files are only read once
per session. Functions return Python objects (DataFrames, dicts) — never HTML.

All paths are relative to PROJECT_ROOT (Byrnihat_air_data/).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR        = PROJECT_ROOT / "data"
PROCESSED_DIR   = DATA_DIR / "processed"
REPORTS_DIR     = PROJECT_ROOT / "outputs" / "reports"
MAPS_DIR        = PROJECT_ROOT / "outputs" / "maps"
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Low-level helpers (not cached — used by cached functions below)
# ---------------------------------------------------------------------------

def _normalize_column_name(column):
    """Lowercase, strip, replace separators — matches existing pipeline convention."""
    column = str(column).strip().lower()
    for char, replacement in [(" ", "_"), (".", "_"), ("-", "_"), ("/", "_"),
                               ("(", ""), (")", "")]:
        column = column.replace(char, replacement)
    return column


def _normalize_column_names(df):
    df = df.copy()
    df.columns = [_normalize_column_name(c) for c in df.columns]
    return df


def _read_csv(path, description="file"):
    """Read a CSV with normalized column names. Raises FileNotFoundError if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found:\n{path}")
    df = pd.read_csv(path)
    df = _normalize_column_names(df)
    return df


def _read_json(path, description="file"):
    """Read a JSON file. Raises FileNotFoundError if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found:\n{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pre-computed prediction outputs
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading prediction output…")
def load_final_predictions():
    """
    Load the main ML-1 + ML-2 prediction output CSV.
    Returns a DataFrame with all recorded and predicted columns.
    Produced by: ml/10_run_final_prediction_pipeline.py
    """
    path = REPORTS_DIR / "final_prediction_output.csv"
    df = _read_csv(path, "Final prediction output")

    # Parse datetime column
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

    return df


@st.cache_data(show_spinner="Loading dashboard graph data…")
def load_graph_data():
    """
    Load the pre-built dashboard graph data (recorded vs predicted series).
    Returns a dict with keys: ml1_recorded_vs_predicted, ml2_simulation_label_vs_prediction,
    error_timeseries, metadata.
    Produced by: ml/10_run_final_prediction_pipeline.py
    """
    path = REPORTS_DIR / "final_dashboard_graph_data.json"
    return _read_json(path, "Dashboard graph data")


@st.cache_data(show_spinner="Loading plume animation frames…")
def load_plume_frames():
    """
    Load the flat plume animation frames CSV.
    Columns: record_id, frame_index, particle_index, lat, lon, size, opacity,
             center_lat, center_lon.
    Produced by: ml/10_run_final_prediction_pipeline.py
    """
    path = MAPS_DIR / "ml_predicted_plume_frames.csv"
    df = _read_csv(path, "Plume frames CSV")
    return df


@st.cache_data(show_spinner="Loading plume simulation data…")
def load_plume_simulation_json():
    """
    Load the full plume simulation JSON (records + frames per record).
    Returns a dict with keys: metadata, records.
    Produced by: ml/10_run_final_prediction_pipeline.py
    """
    path = MAPS_DIR / "ml_predicted_plume_simulation_data.json"
    return _read_json(path, "Plume simulation JSON")


# ---------------------------------------------------------------------------
# ML evaluation outputs
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading ML-1 metrics…")
def load_ml1_metrics():
    """
    Load ML-1 regression metrics (RMSE, MAE, R²) per target pollutant.
    Returns a DataFrame.
    Produced by: ml/07_train_ml1_model.py
    """
    path = REPORTS_DIR / "final_ml1_metrics.csv"
    return _read_csv(path, "ML-1 metrics")


@st.cache_data(show_spinner="Loading ML-2 metrics…")
def load_ml2_metrics():
    """
    Load ML-2 plume direction metrics.
    Returns a DataFrame.
    Produced by: ml/09_train_ml2_plume_direction_model.py
    """
    path = REPORTS_DIR / "final_ml2_metrics.csv"
    return _read_csv(path, "ML-2 metrics")


@st.cache_data(show_spinner="Loading ML-1 feature importance…")
def load_ml1_feature_importance():
    """
    Load ML-1 feature importance scores.
    Returns a DataFrame with columns: feature, importance, target.
    Produced by: ml/07_train_ml1_model.py
    """
    path = REPORTS_DIR / "final_ml1_feature_importance.csv"
    return _read_csv(path, "ML-1 feature importance")


@st.cache_data(show_spinner="Loading ML-2 feature importance…")
def load_ml2_feature_importance():
    """
    Load ML-2 feature importance scores.
    Returns a DataFrame.
    Produced by: ml/09_train_ml2_plume_direction_model.py
    """
    path = REPORTS_DIR / "final_ml2_feature_importance.csv"
    return _read_csv(path, "ML-2 feature importance")


@st.cache_data(show_spinner="Loading ML-1 predictions…")
def load_ml1_predictions():
    """
    Load ML-1 individual predictions CSV (actual vs predicted per row).
    Produced by: ml/07_train_ml1_model.py
    """
    path = REPORTS_DIR / "final_ml1_predictions.csv"
    df = _read_csv(path, "ML-1 predictions")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading ML-2 predictions…")
def load_ml2_predictions():
    """
    Load ML-2 individual predictions CSV (label vs predicted plume vectors).
    Produced by: ml/09_train_ml2_plume_direction_model.py
    """
    path = REPORTS_DIR / "final_ml2_predictions.csv"
    df = _read_csv(path, "ML-2 predictions")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Raw / processed datasets
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading AQICN data…")
def load_aqicn_combined():
    """
    Load the cleaned combined AQICN observed data (all 3 cities).
    Produced by: analysis/observed_data_analysis_v1.py
    """
    path = OUTPUTS_DIR / "observed_aqicn_combined_cleaned.csv"
    df = _read_csv(path, "AQICN combined cleaned")
    for col in ["api_time", "datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading raw city AQICN data…")
def load_raw_city_data():
    """
    Load the 3 individual city AQICN CSV files and return as a combined DataFrame.
    Returns a DataFrame with a 'city' column added.
    """
    city_files = {
        "Byrnihat": DATA_DIR / "byrnihat_aqicn_data.csv",
        "Guwahati": DATA_DIR / "guwahati_aqicn_data.csv",
        "Shillong": DATA_DIR / "shillong_aqicn_data.csv",
    }

    frames = []
    for city_name, fpath in city_files.items():
        if fpath.exists():
            df = pd.read_csv(fpath)
            df = _normalize_column_names(df)
            df["city"] = city_name
            # Standardize time column
            for col in ["api_time", "datetime"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    break
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined


@st.cache_data(show_spinner="Loading ML-1 dataset…")
def load_ml1_dataset():
    """
    Load the ML-1 training dataset (environment + pollution, hourly).
    ~10 MB CSV.
    """
    path = PROCESSED_DIR / "ml1_environment_pollution_dataset.csv"
    df = _read_csv(path, "ML-1 dataset")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading ML-2 dataset…")
def load_ml2_dataset():
    """
    Load the ML-2 plume training dataset (source → grid vectors).
    ~8.5 MB CSV.
    """
    path = PROCESSED_DIR / "ml2_plume_training_dataset.csv"
    df = _read_csv(path, "ML-2 dataset")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading industry sources…")
def load_industry_sources():
    """
    Load active industrial emission sources with coordinates and pollutant weights.
    Uses elevation table if available, falls back to raw source CSV.
    Returns a DataFrame.
    """
    elevation_file = OUTPUTS_DIR / "industry_elevation_table.csv"
    source_file    = DATA_DIR / "byrnihat_33_active_sources_pollutant_weights_updated_raksha_gmaps.csv"

    if elevation_file.exists():
        df = pd.read_csv(elevation_file)
    elif source_file.exists():
        df = pd.read_csv(source_file)
    else:
        return pd.DataFrame(columns=[
            "industry_name", "industry_type", "latitude", "longitude", "pm10_weight"
        ])

    df = _normalize_column_names(df)

    # Keep only active sources
    if "active_for_simulation" in df.columns:
        df = df[df["active_for_simulation"].astype(str).str.lower() == "yes"].copy()

    for col in ["latitude", "longitude"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    return df


@st.cache_data(show_spinner="Loading latest observed values…")
def load_latest_observed():
    """
    Load the latest observed pollution values per city.
    Produced by: analysis/observed_data_analysis_v1.py
    """
    path = OUTPUTS_DIR / "latest_observed_values.csv"
    df = _read_csv(path, "Latest observed values")
    return df


# ---------------------------------------------------------------------------
# File existence checks (for UI status indicators)
# ---------------------------------------------------------------------------

def check_file_status():
    """
    Return a dict of key file statuses (exists / missing) for the dashboard
    data freshness indicator.
    """
    key_files = {
        "Final predictions": REPORTS_DIR / "final_prediction_output.csv",
        "Graph data": REPORTS_DIR / "final_dashboard_graph_data.json",
        "Plume frames": MAPS_DIR / "ml_predicted_plume_frames.csv",
        "ML-1 metrics": REPORTS_DIR / "final_ml1_metrics.csv",
        "ML-2 metrics": REPORTS_DIR / "final_ml2_metrics.csv",
        "ML-1 feature importance": REPORTS_DIR / "final_ml1_feature_importance.csv",
        "ML-2 feature importance": REPORTS_DIR / "final_ml2_feature_importance.csv",
        "ML-1 predictions": REPORTS_DIR / "final_ml1_predictions.csv",
        "ML-2 predictions": REPORTS_DIR / "final_ml2_predictions.csv",
        "Industry sources": OUTPUTS_DIR / "industry_elevation_table.csv",
        "Latest observed": OUTPUTS_DIR / "latest_observed_values.csv",
        "DEM hillshade": MAPS_DIR / "final_terrain_hillshade_overlay.png",
    }

    status = {}
    for label, path in key_files.items():
        exists = Path(path).exists()
        size_mb = round(Path(path).stat().st_size / 1_048_576, 1) if exists else 0
        import time
        age_days = None
        if exists:
            mtime = Path(path).stat().st_mtime
            age_days = round((time.time() - mtime) / 86400, 1)

        status[label] = {
            "exists": exists,
            "path": str(path),
            "size_mb": size_mb,
            "age_days": age_days,
        }

    return status
