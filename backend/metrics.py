"""
backend/metrics.py

Evaluation metric computation for the Byrnihat Streamlit app.
Functions return dicts or DataFrames — never HTML.
Mirrors the metrics used in the original training scripts (sklearn-based).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Regression metrics (ML-1)
# ---------------------------------------------------------------------------

def compute_regression_metrics(y_true, y_pred):
    """
    Compute standard regression metrics between two array-like sequences.

    Returns a dict:
        rmse  : Root Mean Squared Error
        mae   : Mean Absolute Error
        r2    : R² score
        count : Number of valid (non-NaN) pairs used
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Drop rows where either is NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return {"rmse": None, "mae": None, "r2": None, "count": len(y_true)}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    return {
        "rmse":  round(rmse, 4),
        "mae":   round(mae, 4),
        "r2":    round(r2, 4),
        "count": int(len(y_true)),
    }


def compute_metrics_for_all_targets(df, actual_prefix="", predicted_prefix="ml1_predicted_"):
    """
    Compute regression metrics for every actual/predicted pair found in df.

    Looks for columns where:
        - actual column  = <pollutant>   (e.g. "pm25")
        - predicted col  = "ml1_predicted_<pollutant>"

    Returns a DataFrame with columns: target, rmse, mae, r2, count.
    """
    targets = [
        "pm25", "pm10", "so2", "co", "no2", "o3", "nh3", "no", "nox",
        "wind_u", "wind_v",
    ]

    rows = []
    for target in targets:
        actual_col    = actual_prefix + target
        predicted_col = predicted_prefix + target

        if actual_col not in df.columns or predicted_col not in df.columns:
            continue

        m = compute_regression_metrics(df[actual_col], df[predicted_col])
        m["target"] = target
        rows.append(m)

    if not rows:
        return pd.DataFrame(columns=["target", "rmse", "mae", "r2", "count"])

    result_df = pd.DataFrame(rows)[["target", "rmse", "mae", "r2", "count"]]
    return result_df


# ---------------------------------------------------------------------------
# Angular / directional metrics (ML-2 plume direction)
# ---------------------------------------------------------------------------

def compute_angular_error(y_true_angle, y_pred_angle):
    """
    Compute the circular angular error between two angle series (degrees).
    Handles wraparound at 0°/360°.

    Returns the minimum absolute angular difference (0–180°).
    """
    y_true = np.array(y_true_angle, dtype=float)
    y_pred = np.array(y_pred_angle, dtype=float)

    diff = np.abs(y_pred - y_true) % 360
    angular_error = np.minimum(diff, 360 - diff)

    return angular_error


def compute_angular_metrics(df):
    """
    Compute ML-2 plume direction metrics from the final predictions DataFrame.

    Expects columns:
        simulation_label_plume_angle   (ground-truth angle in degrees)
        ml2_predicted_plume_angle      (predicted angle in degrees)
        final_angular_error            (pre-computed, optional)
        final_vector_error             (pre-computed, optional)

    Returns a dict:
        mean_angular_error_deg   : average angular error in degrees
        median_angular_error_deg : median angular error
        pct_within_30_deg        : % predictions within 30° of truth
        pct_within_45_deg        : % predictions within 45° of truth
        mean_vector_error        : mean Euclidean vector error (U,V space)
        count                    : number of valid rows
    """
    result = {
        "mean_angular_error_deg":   None,
        "median_angular_error_deg": None,
        "pct_within_30_deg":        None,
        "pct_within_45_deg":        None,
        "mean_vector_error":        None,
        "count":                    0,
    }

    # Use pre-computed column if available
    if "final_angular_error" in df.columns:
        angular_error = pd.to_numeric(df["final_angular_error"], errors="coerce").dropna()
    elif ("simulation_label_plume_angle" in df.columns
          and "ml2_predicted_plume_angle" in df.columns):
        angular_error = pd.Series(
            compute_angular_error(
                df["simulation_label_plume_angle"],
                df["ml2_predicted_plume_angle"],
            )
        )
        angular_error = angular_error.dropna()
    else:
        return result

    if len(angular_error) == 0:
        return result

    result["mean_angular_error_deg"]   = round(float(angular_error.mean()), 2)
    result["median_angular_error_deg"] = round(float(angular_error.median()), 2)
    result["pct_within_30_deg"]        = round(float((angular_error <= 30).mean() * 100), 1)
    result["pct_within_45_deg"]        = round(float((angular_error <= 45).mean() * 100), 1)
    result["count"]                    = int(len(angular_error))

    if "final_vector_error" in df.columns:
        vec_err = pd.to_numeric(df["final_vector_error"], errors="coerce").dropna()
        if len(vec_err) > 0:
            result["mean_vector_error"] = round(float(vec_err.mean()), 4)

    return result


# ---------------------------------------------------------------------------
# Plume strength metrics (ML-2)
# ---------------------------------------------------------------------------

def compute_plume_strength_metrics(df):
    """
    Compute ML-2 plume strength regression metrics.

    Expects columns:
        simulation_label_plume_strength
        ml2_predicted_plume_strength

    Returns a dict with rmse, mae, r2.
    """
    if ("simulation_label_plume_strength" not in df.columns
            or "ml2_predicted_plume_strength" not in df.columns):
        return {}

    return compute_regression_metrics(
        df["simulation_label_plume_strength"],
        df["ml2_predicted_plume_strength"],
    )


# ---------------------------------------------------------------------------
# Summary metrics for the Dashboard page
# ---------------------------------------------------------------------------

def get_dashboard_summary_metrics(final_df):
    """
    Extract a compact set of key metrics for the Dashboard overview panel.

    Returns a dict:
        ml1_pm25_rmse   : ML-1 PM2.5 RMSE
        ml1_pm10_rmse   : ML-1 PM10 RMSE
        ml2_angular_err : Mean plume angular error (degrees)
        ml2_pct_30      : % predictions within 30°
        total_records   : total row count in final_df
        date_range      : (min_date, max_date) tuple or (None, None)
    """
    summary = {
        "ml1_pm25_rmse":   None,
        "ml1_pm10_rmse":   None,
        "ml2_angular_err": None,
        "ml2_pct_30":      None,
        "total_records":   len(final_df),
        "date_range":      (None, None),
    }

    # ML-1 quick metrics
    if "pm25" in final_df.columns and "ml1_predicted_pm25" in final_df.columns:
        m = compute_regression_metrics(final_df["pm25"], final_df["ml1_predicted_pm25"])
        summary["ml1_pm25_rmse"] = m["rmse"]

    if "pm10" in final_df.columns and "ml1_predicted_pm10" in final_df.columns:
        m = compute_regression_metrics(final_df["pm10"], final_df["ml1_predicted_pm10"])
        summary["ml1_pm10_rmse"] = m["rmse"]

    # ML-2 angular metrics
    angular = compute_angular_metrics(final_df)
    summary["ml2_angular_err"] = angular["mean_angular_error_deg"]
    summary["ml2_pct_30"]      = angular["pct_within_30_deg"]

    # Date range
    if "datetime" in final_df.columns:
        dt = pd.to_datetime(final_df["datetime"], errors="coerce").dropna()
        if len(dt) > 0:
            summary["date_range"] = (dt.min(), dt.max())

    return summary
