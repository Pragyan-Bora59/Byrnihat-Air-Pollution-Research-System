"""
pages/3_ML_Predictions.py — ML Predictions

Displays:
  ML-1: PM2.5/PM10/weather recorded vs predicted time series and scatter
  ML-2: Plume direction comparison, angular error, wind rose

Models are NOT loaded or retrained here — only pre-computed outputs are shown.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data_loader    import load_final_predictions, load_graph_data
from backend.chart_builder  import (
    plot_ml1_actual_vs_predicted_line,
    plot_ml1_scatter_actual_vs_predicted,
    plot_ml1_residuals,
    plot_ml2_plume_angle_comparison,
    plot_ml2_error_timeseries,
    plot_wind_rose,
)
from backend.prediction_engine import (
    get_latest_ml1_predictions,
    get_latest_ml2_predictions,
    get_pollution_columns,
    get_date_filtered_predictions,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ML Predictions — Byrnihat",
    page_icon="🤖",
    layout="wide",
)

CSS = Path(__file__).parents[1] / "assets" / "style.css"
if CSS.exists():
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

st.markdown("## 🤖 ML Predictions")
st.markdown(
    "<div style='color:#94a3b8; margin-bottom:1.5rem'>"
    "Pre-computed ML-1 pollution/weather predictions and ML-2 plume direction results. "
    "No models are loaded or retrained here."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

try:
    pred_df    = load_final_predictions()
    graph_data = load_graph_data()
    data_ok    = True
except FileNotFoundError as e:
    pred_df    = pd.DataFrame()
    graph_data = {}
    data_ok    = False
    st.error(f"⚠️ Data not found: {e}\n\nRun `python ml/10_run_final_prediction_pipeline.py` first.")

if not data_ok:
    st.stop()


# ---------------------------------------------------------------------------
# Date range filter (shared between tabs)
# ---------------------------------------------------------------------------

if "datetime" in pred_df.columns:
    pred_df["datetime"] = pd.to_datetime(pred_df["datetime"], errors="coerce")
    min_dt = pred_df["datetime"].min().date()
    max_dt = pred_df["datetime"].max().date()
else:
    min_dt = max_dt = None

with st.expander("🗓️ Date Range Filter", expanded=False):
    if min_dt and max_dt:
        date_cols = st.columns(2)
        with date_cols[0]:
            start_date = st.date_input("From", value=min_dt, min_value=min_dt,
                                       max_value=max_dt, key="ml_start")
        with date_cols[1]:
            end_date   = st.date_input("To",   value=max_dt, min_value=min_dt,
                                       max_value=max_dt, key="ml_end")
        pred_df = get_date_filtered_predictions(pred_df, start_date, end_date)
        st.caption(f"Showing {len(pred_df):,} rows from {start_date} to {end_date}")
    else:
        st.info("No datetime column available for filtering.")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["🌫️ ML-1 Pollution & Weather", "🧭 ML-2 Plume Direction"])


# ============================================================
# TAB 1 — ML-1
# ============================================================

with tab1:
    st.markdown("### ML-1: Pollution & Weather Predictions")
    st.markdown(
        "<div style='color:#94a3b8; font-size:0.9rem; margin-bottom:1rem'>"
        "ExtraTreesRegressor predictions for PM2.5, PM10, SO2, CO, NO2, O3, NH3, NO, NOx, "
        "Wind U, Wind V. Select a target below to explore recorded vs predicted values."
        "</div>",
        unsafe_allow_html=True,
    )

    # Target selector
    pollutant_options = {
        "pm25":    "PM2.5 (µg/m³)",
        "pm10":    "PM10 (µg/m³)",
        "so2":     "SO2 (µg/m³)",
        "co":      "CO (mg/m³)",
        "no2":     "NO2 (µg/m³)",
        "o3":      "O3 (µg/m³)",
        "nh3":     "NH3 (µg/m³)",
        "no":      "NO (µg/m³)",
        "nox":     "NOx (µg/m³)",
        "wind_u":  "Wind U (m/s)",
        "wind_v":  "Wind V (m/s)",
    }

    # Filter to only targets present in data
    available = [k for k in pollutant_options
                 if f"ml1_predicted_{k}" in pred_df.columns]
    selected_target = st.selectbox(
        "Select pollutant / weather target",
        options=available,
        format_func=lambda k: pollutant_options.get(k, k),
        key="ml1_target",
    )

    v1, v2 = st.columns(2)
    with v1:
        chart_type = st.radio(
            "Chart type",
            ["Time Series", "Scatter (Actual vs Predicted)", "Residuals"],
            horizontal=True,
            key="ml1_chart_type",
        )
    with v2:
        show_table = st.checkbox("Show prediction table", value=False, key="ml1_show_table")

    # Chart
    if chart_type == "Time Series":
        fig = plot_ml1_actual_vs_predicted_line(graph_data, selected_target)
    elif chart_type == "Scatter (Actual vs Predicted)":
        fig = plot_ml1_scatter_actual_vs_predicted(pred_df, selected_target)
    else:
        fig = plot_ml1_residuals(pred_df, selected_target)

    st.plotly_chart(fig, use_container_width=True)

    # Quick metrics below chart
    if selected_target in pred_df.columns and f"ml1_predicted_{selected_target}" in pred_df.columns:
        from backend.metrics import compute_regression_metrics
        valid = pred_df[[selected_target, f"ml1_predicted_{selected_target}"]].dropna()
        if not valid.empty:
            m = compute_regression_metrics(
                valid[selected_target],
                valid[f"ml1_predicted_{selected_target}"],
            )
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("RMSE", f"{m['rmse']:.4f}" if m["rmse"] else "N/A")
            with mc2:
                st.metric("MAE",  f"{m['mae']:.4f}"  if m["mae"]  else "N/A")
            with mc3:
                st.metric("R²",   f"{m['r2']:.4f}"   if m["r2"]   else "N/A")
            with mc4:
                st.metric("Valid Pairs", f"{m['count']:,}")

    # Optional prediction table
    if show_table:
        table_df = get_latest_ml1_predictions(pred_df, n=200)
        # Show only relevant columns
        show_cols = ["datetime"] if "datetime" in table_df.columns else []
        show_cols += [selected_target, f"ml1_predicted_{selected_target}"]
        show_cols = [c for c in show_cols if c in table_df.columns]
        if show_cols:
            st.dataframe(
                table_df[show_cols].tail(100).round(4),
                use_container_width=True,
                hide_index=True,
            )
        st.download_button(
            "⬇️ Download ML-1 predictions (200 rows)",
            data=get_latest_ml1_predictions(pred_df, n=200).to_csv(index=False),
            file_name=f"ml1_predictions_{selected_target}.csv",
            mime="text/csv",
            key="ml1_download",
        )

    # Wind rose (shown when wind target is selected)
    if selected_target in ("wind_u", "wind_v"):
        st.markdown("---")
        st.markdown("**🌪️ Wind Rose**")
        fig_rose = plot_wind_rose(pred_df)
        st.plotly_chart(fig_rose, use_container_width=True)


# ============================================================
# TAB 2 — ML-2
# ============================================================

with tab2:
    st.markdown("### ML-2: Plume Direction Prediction")
    st.markdown(
        "<div style='color:#94a3b8; font-size:0.9rem; margin-bottom:1rem'>"
        "Multi-output ExtraTreesRegressor predicting plume U/V vectors → "
        "angle and strength. Compared against simulation-derived labels."
        "</div>",
        unsafe_allow_html=True,
    )

    p1, p2 = st.columns(2)

    with p1:
        st.markdown("#### Plume Angle: Label vs Predicted")
        fig_angle = plot_ml2_plume_angle_comparison(graph_data)
        st.plotly_chart(fig_angle, use_container_width=True)

    with p2:
        st.markdown("#### Prediction Error Over Time")
        fig_err = plot_ml2_error_timeseries(graph_data)
        st.plotly_chart(fig_err, use_container_width=True)

    # Angular metrics summary
    st.markdown("---")
    st.markdown("#### Angular & Vector Metrics")

    from backend.metrics import compute_angular_metrics, compute_plume_strength_metrics
    ang_m = compute_angular_metrics(pred_df)
    str_m = compute_plume_strength_metrics(pred_df)

    am1, am2, am3, am4 = st.columns(4)
    with am1:
        val = f"{ang_m['mean_angular_error_deg']:.2f}°" if ang_m["mean_angular_error_deg"] else "N/A"
        st.metric("Mean Angular Error", val)
    with am2:
        val = f"{ang_m['median_angular_error_deg']:.2f}°" if ang_m["median_angular_error_deg"] else "N/A"
        st.metric("Median Angular Error", val)
    with am3:
        val = f"{ang_m['pct_within_30_deg']:.1f}%" if ang_m["pct_within_30_deg"] else "N/A"
        st.metric("Within 30°", val)
    with am4:
        val = f"{ang_m['pct_within_45_deg']:.1f}%" if ang_m["pct_within_45_deg"] else "N/A"
        st.metric("Within 45°", val)

    if str_m:
        sm1, sm2, sm3 = st.columns(3)
        with sm1:
            st.metric("Plume Strength RMSE", f"{str_m.get('rmse', 'N/A')}")
        with sm2:
            st.metric("Plume Strength MAE",  f"{str_m.get('mae', 'N/A')}")
        with sm3:
            st.metric("Plume Strength R²",   f"{str_m.get('r2', 'N/A')}")

    # ML-2 prediction table
    with st.expander("📋 ML-2 Prediction Table (last 100 records)", expanded=False):
        ml2_table = get_latest_ml2_predictions(pred_df, n=100)
        if not ml2_table.empty:
            st.dataframe(ml2_table.round(4), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download ML-2 predictions",
                data=ml2_table.to_csv(index=False),
                file_name="ml2_predictions.csv",
                mime="text/csv",
                key="ml2_download",
            )
