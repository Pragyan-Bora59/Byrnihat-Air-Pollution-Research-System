"""
backend/prediction_engine.py

Thin wrapper around the pre-computed prediction outputs.
The Streamlit app NEVER retrains models or re-runs the pipeline.
All functions load from outputs/reports/ and outputs/maps/ directories.

If a pipeline re-run is needed, the user should run:
    python ml/10_run_final_prediction_pipeline.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from backend.data_loader import (
    load_final_predictions,
    load_graph_data,
    load_ml1_metrics,
    load_ml2_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR  = PROJECT_ROOT / "outputs" / "reports"
MAPS_DIR     = PROJECT_ROOT / "outputs" / "maps"


# ---------------------------------------------------------------------------
# Key pre-computed file paths
# ---------------------------------------------------------------------------

PREDICTION_CSV      = REPORTS_DIR / "final_prediction_output.csv"
GRAPH_DATA_JSON     = REPORTS_DIR / "final_dashboard_graph_data.json"
ML1_METRICS_CSV     = REPORTS_DIR / "final_ml1_metrics.csv"
ML2_METRICS_CSV     = REPORTS_DIR / "final_ml2_metrics.csv"
PLUME_FRAMES_CSV    = MAPS_DIR    / "ml_predicted_plume_frames.csv"
PIPELINE_REPORT_JSON = REPORTS_DIR / "final_prediction_pipeline_report.json"


# ---------------------------------------------------------------------------
# Pre-computed output accessors
# ---------------------------------------------------------------------------

def get_final_predictions():
    """
    Return the full ML-1 + ML-2 prediction output DataFrame.
    Loaded from outputs/reports/final_prediction_output.csv.
    (Cached internally by data_loader.load_final_predictions)
    """
    return load_final_predictions()


def get_latest_ml1_predictions(df=None, n=100):
    """
    Return the N most recent rows of the final predictions DataFrame,
    focusing on ML-1 columns (pollution + weather predictions).

    Useful for the ML Predictions page summary table.
    """
    if df is None:
        df = load_final_predictions()

    ml1_cols = [c for c in df.columns if c.startswith("ml1_predicted_")]
    base_cols = ["datetime", "station_name", "source_name"]
    base_cols = [c for c in base_cols if c in df.columns]

    keep_cols = base_cols + ml1_cols
    result = df[keep_cols].tail(n).copy()

    return result


def get_latest_ml2_predictions(df=None, n=100):
    """
    Return the N most recent rows focusing on ML-2 columns (plume direction).
    """
    if df is None:
        df = load_final_predictions()

    ml2_cols = [c for c in df.columns if c.startswith("ml2_predicted_")]
    extra_cols = ["simulation_label_plume_angle", "simulation_label_plume_strength",
                  "final_angular_error", "final_vector_error",
                  "target_plume_u", "target_plume_v"]
    base_cols  = ["datetime", "station_name", "source_name", "source_lat", "source_lon"]

    keep_cols = (
        [c for c in base_cols if c in df.columns]
        + ml2_cols
        + [c for c in extra_cols if c in df.columns]
    )
    result = df[keep_cols].tail(n).copy()

    return result


def get_pollution_columns(df=None):
    """Return list of actual pollution columns present in the predictions DataFrame."""
    if df is None:
        df = load_final_predictions()
    candidates = ["pm25", "pm10", "so2", "co", "no2", "o3", "nh3", "no", "nox"]
    return [c for c in candidates if c in df.columns]


def get_ml1_predicted_columns(df=None):
    """Return list of ML-1 predicted column names present in the predictions DataFrame."""
    if df is None:
        df = load_final_predictions()
    return [c for c in df.columns if c.startswith("ml1_predicted_")]


def get_dashboard_snapshot(df=None):
    """
    Return a compact dict of latest observed and predicted values
    for the Dashboard page metric cards.

    Returns a dict:
        latest_datetime : most recent datetime
        latest_pm25     : latest recorded PM2.5
        latest_pm10     : latest recorded PM10
        pred_pm25       : latest ML-1 predicted PM2.5
        pred_pm10       : latest ML-1 predicted PM10
        plume_angle     : latest ML-2 predicted plume angle (°)
        plume_strength  : latest ML-2 predicted plume strength
        wind_speed      : latest recorded wind speed
        wind_direction  : latest recorded wind direction
        temperature     : latest recorded temperature
        humidity        : latest recorded humidity
    """
    if df is None:
        df = load_final_predictions()

    snapshot = {
        "latest_datetime":  None,
        "latest_pm25":      None,
        "latest_pm10":      None,
        "pred_pm25":        None,
        "pred_pm10":        None,
        "plume_angle":      None,
        "plume_strength":   None,
        "wind_speed":       None,
        "wind_direction":   None,
        "temperature":      None,
        "humidity":         None,
    }

    if df.empty:
        return snapshot

    # Use the last non-null row for each column
    last = df.iloc[-1]

    def _get(col):
        val = last.get(col) if col in df.columns else None
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return round(float(val), 2) if isinstance(val, (float, int, np.floating)) else val
        return None

    snapshot["latest_datetime"] = _get("datetime")
    snapshot["latest_pm25"]     = _get("pm25")
    snapshot["latest_pm10"]     = _get("pm10")
    snapshot["pred_pm25"]       = _get("ml1_predicted_pm25")
    snapshot["pred_pm10"]       = _get("ml1_predicted_pm10")
    snapshot["plume_angle"]     = _get("ml2_predicted_plume_angle")
    snapshot["plume_strength"]  = _get("ml2_predicted_plume_strength")
    snapshot["wind_speed"]      = _get("wind_speed")
    snapshot["wind_direction"]  = _get("wind_direction")
    snapshot["temperature"]     = _get("temperature")
    snapshot["humidity"]        = _get("humidity")

    return snapshot


def check_pipeline_outputs_exist():
    """
    Check which prediction pipeline output files are present.
    Returns a dict of {filename: True/False}.
    """
    files = {
        "final_prediction_output.csv":          PREDICTION_CSV,
        "final_dashboard_graph_data.json":       GRAPH_DATA_JSON,
        "final_ml1_metrics.csv":                 ML1_METRICS_CSV,
        "final_ml2_metrics.csv":                 ML2_METRICS_CSV,
        "ml_predicted_plume_frames.csv":         PLUME_FRAMES_CSV,
        "final_prediction_pipeline_report.json": PIPELINE_REPORT_JSON,
    }
    return {name: Path(path).exists() for name, path in files.items()}


def get_pipeline_report():
    """
    Load and return the pipeline execution report dict.
    Contains merge keys, ML-1 prediction columns, final row count, etc.
    """
    import json
    if not PIPELINE_REPORT_JSON.exists():
        return {}
    with open(PIPELINE_REPORT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_date_filtered_predictions(df, start_date=None, end_date=None):
    """
    Filter the predictions DataFrame to a date range.
    start_date / end_date: strings or datetime-like, inclusive.
    """
    if "datetime" not in df.columns:
        return df

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    if start_date is not None:
        df = df[df["datetime"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["datetime"] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)
