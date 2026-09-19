"""
pages/1_Dashboard.py — Project Dashboard

Displays:
  - Latest AQI / PM2.5 / PM10 per city
  - Model performance summary metrics
  - Recent pollution trend sparklines
  - Current prediction snapshot
  - Data freshness alerts
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make backend importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.analysis_engine import (
    get_aqi_category,
    get_latest_values,
    load_all_city_data,
    get_pollution_trend,
)
from backend.data_loader     import load_final_predictions, load_latest_observed
from backend.metrics         import get_dashboard_summary_metrics
from backend.prediction_engine import get_dashboard_snapshot


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Dashboard — Byrnihat",
    page_icon="📊",
    layout="wide",
)

# Inject CSS
CSS = Path(__file__).parents[1] / "assets" / "style.css"
if CSS.exists():
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

st.markdown("## 📊 Dashboard")
st.markdown(
    "<div style='color:#94a3b8; margin-bottom:1.5rem'>"
    "Latest air quality metrics, model performance summary and system status."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_city_data():
    return load_all_city_data()


@st.cache_data(show_spinner=False)
def _load_predictions():
    return load_final_predictions()


with st.spinner("Loading dashboard data…"):
    city_df = _load_city_data()
    try:
        pred_df = _load_predictions()
        pred_ok = True
    except FileNotFoundError as e:
        pred_df = pd.DataFrame()
        pred_ok = False
        st.error(f"⚠️ Prediction output not found: {e}\n\n"
                 f"Run `python ml/10_run_final_prediction_pipeline.py` first.")


# ---------------------------------------------------------------------------
# Section 1: City AQI status cards
# ---------------------------------------------------------------------------

st.markdown(
    "<div class='section-header'>🏙️ City Air Quality Status</div>",
    unsafe_allow_html=True,
)

cities     = ["Byrnihat", "Guwahati", "Shillong"]
time_col   = "api_time" if "api_time" in city_df.columns else "datetime"
city_cols  = st.columns(3)

for col, city in zip(city_cols, cities):
    with col:
        if not city_df.empty and "city" in city_df.columns:
            city_data = (city_df[city_df["city"] == city]
                         .sort_values(time_col)
                         .dropna(subset=[time_col]))
        else:
            city_data = pd.DataFrame()

        if city_data.empty:
            st.metric(label=f"🏙️ {city}", value="No data", delta=None)
            continue

        latest = city_data.iloc[-1]
        aqi    = latest.get("aqi") if "aqi" in city_data.columns else None
        pm25   = latest.get("pm25") if "pm25" in city_data.columns else None
        pm10   = latest.get("pm10") if "pm10" in city_data.columns else None
        ts     = latest.get(time_col)

        cat, colour = get_aqi_category(aqi)

        # Sparkline — last 10 daily AQI readings
        spark_df = get_pollution_trend(city_df, "aqi", city, last_n=10)

        aqi_str  = f"{aqi:.0f}" if aqi is not None else "N/A"
        pm25_str = f"{pm25:.1f}" if pm25 is not None else "N/A"
        pm10_str = f"{pm10:.1f}" if pm10 is not None else "N/A"
        ts_str   = str(ts)[:16] if ts is not None else "Unknown"

        st.markdown(
            f"""
            <div style='background:#1e293b; border:1px solid #334155;
                        border-radius:12px; padding:1.2rem; margin-bottom:0.5rem'>
                <div style='display:flex; justify-content:space-between;
                            align-items:center; margin-bottom:0.6rem'>
                    <span style='font-size:1.05rem; font-weight:700;
                                 color:#f1f5f9'>🏙️ {city}</span>
                    <span class='aqi-badge'
                          style='background:{colour}22; color:{colour};
                                 border:1px solid {colour}44'>
                        {cat}
                    </span>
                </div>
                <div style='display:grid; grid-template-columns:1fr 1fr 1fr;
                            gap:0.5rem; margin-bottom:0.5rem'>
                    <div>
                        <div style='font-size:0.7rem; color:#94a3b8'>AQI</div>
                        <div style='font-size:1.4rem; font-weight:700;
                                    color:{colour}'>{aqi_str}</div>
                    </div>
                    <div>
                        <div style='font-size:0.7rem; color:#94a3b8'>PM2.5</div>
                        <div style='font-size:1.4rem; font-weight:700;
                                    color:#f1f5f9'>{pm25_str}</div>
                    </div>
                    <div>
                        <div style='font-size:0.7rem; color:#94a3b8'>PM10</div>
                        <div style='font-size:1.4rem; font-weight:700;
                                    color:#f1f5f9'>{pm10_str}</div>
                    </div>
                </div>
                <div style='font-size:0.72rem; color:#475569'>
                    Last updated: {ts_str}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Sparkline chart
        if not spark_df.empty:
            spark_time = spark_df.columns[0] if spark_df.columns[0] != "aqi" else time_col
            spark_y    = spark_df.get("aqi", spark_df.iloc[:, -1])
            fig_spark  = go.Figure(go.Scatter(
                y=spark_y, mode="lines+markers",
                line=dict(color=colour, width=2),
                marker=dict(size=4),
                fill="tozeroy",
                fillcolor=f"rgba(96,165,250,0.08)",
            ))
            fig_spark.update_layout(
                height=80, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                showlegend=False,
            )
            st.plotly_chart(fig_spark, use_container_width=True,
                            config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Section 2: ML model performance summary
# ---------------------------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-header'>🤖 Model Performance Summary</div>",
    unsafe_allow_html=True,
)

if pred_ok and not pred_df.empty:
    summary = get_dashboard_summary_metrics(pred_df)

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        val = f"{summary['ml1_pm25_rmse']:.2f}" if summary["ml1_pm25_rmse"] else "N/A"
        st.metric("ML-1 PM2.5 RMSE", val, help="Root Mean Squared Error for PM2.5 prediction")
    with m2:
        val = f"{summary['ml1_pm10_rmse']:.2f}" if summary["ml1_pm10_rmse"] else "N/A"
        st.metric("ML-1 PM10 RMSE", val, help="Root Mean Squared Error for PM10 prediction")
    with m3:
        val = f"{summary['ml2_angular_err']:.1f}°" if summary["ml2_angular_err"] else "N/A"
        st.metric("ML-2 Mean Angular Error", val,
                  help="Mean circular angular error of plume direction prediction")
    with m4:
        val = f"{summary['ml2_pct_30']:.1f}%" if summary["ml2_pct_30"] else "N/A"
        st.metric("Predictions within 30°", val,
                  help="% of plume direction predictions within 30° of truth")
    with m5:
        val = f"{summary['total_records']:,}"
        st.metric("Total Records", val,
                  help="Total rows in the final prediction output")

    # Date range
    if summary["date_range"][0] is not None:
        d_min = str(summary["date_range"][0])[:10]
        d_max = str(summary["date_range"][1])[:10]
        st.caption(f"📅 Prediction data spans: **{d_min}** to **{d_max}**")
else:
    st.info("Run the prediction pipeline to populate model performance metrics.")


# ---------------------------------------------------------------------------
# Section 3: Current prediction snapshot
# ---------------------------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-header'>🔭 Latest Prediction Snapshot</div>",
    unsafe_allow_html=True,
)

if pred_ok and not pred_df.empty:
    snap = get_dashboard_snapshot(pred_df)

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown("**🌡️ Weather**")
        temp  = f"{snap['temperature']:.1f} °C"   if snap["temperature"]  else "N/A"
        hum   = f"{snap['humidity']:.1f} %"        if snap["humidity"]     else "N/A"
        ws    = f"{snap['wind_speed']:.2f} m/s"    if snap["wind_speed"]   else "N/A"
        wd    = f"{snap['wind_direction']:.1f} °"  if snap["wind_direction"] else "N/A"
        st.markdown(f"- Temperature: **{temp}**")
        st.markdown(f"- Humidity: **{hum}**")
        st.markdown(f"- Wind Speed: **{ws}**")
        st.markdown(f"- Wind Direction: **{wd}**")

    with s2:
        st.markdown("**💨 ML-1 Predictions**")
        p25  = f"{snap['pred_pm25']:.2f} µg/m³"   if snap["pred_pm25"]      else "N/A"
        p10  = f"{snap['pred_pm10']:.2f} µg/m³"   if snap["pred_pm10"]      else "N/A"
        r25  = f"{snap['latest_pm25']:.2f} µg/m³" if snap["latest_pm25"]   else "N/A"
        r10  = f"{snap['latest_pm10']:.2f} µg/m³" if snap["latest_pm10"]   else "N/A"
        st.markdown(f"- Predicted PM2.5: **{p25}**")
        st.markdown(f"- Recorded PM2.5: **{r25}**")
        st.markdown(f"- Predicted PM10: **{p10}**")
        st.markdown(f"- Recorded PM10: **{r10}**")

    with s3:
        st.markdown("**🌫️ ML-2 Plume Direction**")
        angle  = f"{snap['plume_angle']:.1f} °"   if snap["plume_angle"]    else "N/A"
        strength = f"{snap['plume_strength']:.3f}" if snap["plume_strength"] else "N/A"
        ts     = str(snap["latest_datetime"])[:19]  if snap["latest_datetime"] else "N/A"
        st.markdown(f"- Predicted Plume Angle: **{angle}**")
        st.markdown(f"- Plume Strength (vector): **{strength}**")
        st.markdown(f"- Last record datetime: **{ts}**")
else:
    st.info("No prediction data available. "
            "Run `python ml/10_run_final_prediction_pipeline.py` to generate it.")


# ---------------------------------------------------------------------------
# Section 4: Recent pollution trend (last 14 days from predictions)
# ---------------------------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-header'>📈 Recent Pollution Trend (Predictions)</div>",
    unsafe_allow_html=True,
)

if pred_ok and not pred_df.empty and "datetime" in pred_df.columns:
    # Get last 14 days
    pred_df_copy = pred_df.copy()
    pred_df_copy["datetime"] = pd.to_datetime(pred_df_copy["datetime"], errors="coerce")
    recent = pred_df_copy.sort_values("datetime").tail(24 * 14)  # ~14 days of hourly

    plot_cols = [c for c in ["pm25", "pm10", "ml1_predicted_pm25", "ml1_predicted_pm10"]
                 if c in recent.columns]

    if plot_cols:
        import plotly.graph_objects as go
        fig = go.Figure()
        colours = {"pm25": "#60a5fa", "pm10": "#f97316",
                   "ml1_predicted_pm25": "#34d399", "ml1_predicted_pm10": "#a78bfa"}
        names   = {"pm25": "PM2.5 Recorded", "pm10": "PM10 Recorded",
                   "ml1_predicted_pm25": "PM2.5 Predicted", "ml1_predicted_pm10": "PM10 Predicted"}

        for c in plot_cols:
            fig.add_trace(go.Scatter(
                x=recent["datetime"], y=recent[c],
                name=names.get(c, c),
                mode="lines",
                line=dict(color=colours.get(c, "#94a3b8"), width=1.5),
                opacity=0.85,
            ))

        fig.update_layout(
            height=320,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Datetime",
            yaxis_title="µg/m³",
            legend=dict(orientation="h", y=1.08),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No PM2.5/PM10 columns found in prediction output.")
else:
    st.info("Load the prediction output to see recent trend chart.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<div style='font-size:0.78rem; color:#475569'>"
    "Data shown is from pre-computed pipeline outputs in "
    "<code>outputs/reports/</code> and <code>outputs/maps/</code>. "
    "Re-run <code>ml/10_run_final_prediction_pipeline.py</code> to refresh."
    "</div>",
    unsafe_allow_html=True,
)
