"""
pages/4_Historical_Analysis.py — Historical Analysis

Displays:
  - Multi-city pollution time series (Byrnihat, Guwahati, Shillong)
  - Date range filtering
  - Monthly distribution box plots
  - Yearly trend lines
  - Summary statistics table
  - Download filtered data
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data_loader    import load_aqicn_combined, load_raw_city_data
from backend.analysis_engine import (
    compute_city_summary,
    compute_missing_report,
    get_latest_values,
)
from backend.chart_builder  import (
    plot_pollution_timeseries,
    plot_monthly_stats,
    plot_yearly_trend,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Historical Analysis — Byrnihat",
    page_icon="📈",
    layout="wide",
)

CSS = Path(__file__).parents[1] / "assets" / "style.css"
if CSS.exists():
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

st.markdown("## 📈 Historical Analysis")
st.markdown(
    "<div style='color:#94a3b8; margin-bottom:1.5rem'>"
    "Observed AQICN pollution data for Byrnihat, Guwahati and Shillong — "
    "time series, monthly statistics, yearly trends and summary tables."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading historical AQICN data…")
def _load_historical():
    try:
        df = load_aqicn_combined()
        if df.empty:
            df = load_raw_city_data()
        return df
    except Exception:
        return load_raw_city_data()


with st.spinner("Loading historical data…"):
    hist_df = _load_historical()

if hist_df.empty:
    st.warning("No historical AQICN data found in `data/` or `outputs/`.")
    st.stop()

# Detect time column
time_col = "api_time" if "api_time" in hist_df.columns else "datetime"
if time_col in hist_df.columns:
    hist_df[time_col] = pd.to_datetime(hist_df[time_col], errors="coerce")
    hist_df = hist_df.dropna(subset=[time_col])


# ---------------------------------------------------------------------------
# Sidebar-style filters (using columns)
# ---------------------------------------------------------------------------

filt1, filt2, filt3 = st.columns([2, 2, 3])

with filt1:
    all_cities = sorted(hist_df["city"].unique().tolist()) if "city" in hist_df.columns else []
    selected_cities = st.multiselect(
        "Cities",
        options=all_cities,
        default=all_cities,
        key="hist_cities",
    )

with filt2:
    pollutant_opts = {
        "aqi":  "AQI",
        "pm25": "PM2.5",
        "pm10": "PM10",
        "no2":  "NO2",
        "so2":  "SO2",
        "o3":   "O3",
        "co":   "CO",
    }
    available_poll = [k for k in pollutant_opts if k in hist_df.columns]
    selected_polls = st.multiselect(
        "Pollutants",
        options=available_poll,
        default=available_poll[:3],
        format_func=lambda k: pollutant_opts[k],
        key="hist_polls",
    )

with filt3:
    if time_col in hist_df.columns and len(hist_df) > 0:
        min_dt = hist_df[time_col].min().date()
        max_dt = hist_df[time_col].max().date()
        d1, d2 = st.columns(2)
        with d1:
            start_d = st.date_input("From", value=min_dt, min_value=min_dt,
                                    max_value=max_dt, key="hist_start")
        with d2:
            end_d   = st.date_input("To",   value=max_dt, min_value=min_dt,
                                    max_value=max_dt, key="hist_end")

# Apply filters
filtered = hist_df.copy()
if selected_cities and "city" in filtered.columns:
    filtered = filtered[filtered["city"].isin(selected_cities)]
if time_col in filtered.columns:
    filtered = filtered[
        (filtered[time_col].dt.date >= start_d) &
        (filtered[time_col].dt.date <= end_d)
    ]

st.caption(f"Showing **{len(filtered):,}** rows | "
           f"{len(selected_cities)} cities | "
           f"{len(selected_polls)} pollutants")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Time Series",
    "📦 Monthly Distribution",
    "📆 Yearly Trend",
    "📋 Summary Statistics",
])


# ── Tab 1: Time Series ──────────────────────────────────────

with tab1:
    if not selected_polls:
        st.info("Select at least one pollutant above.")
    elif filtered.empty:
        st.warning("No data for the selected filters.")
    else:
        fig_ts = plot_pollution_timeseries(
            filtered,
            pollutants=selected_polls,
            city_col="city",
            time_col=time_col,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    st.download_button(
        "⬇️ Download filtered data (CSV)",
        data=filtered.to_csv(index=False),
        file_name="historical_pollution_filtered.csv",
        mime="text/csv",
        key="hist_dl_ts",
    )


# ── Tab 2: Monthly Distribution ─────────────────────────────

with tab2:
    if not selected_polls:
        st.info("Select at least one pollutant above.")
    elif filtered.empty:
        st.warning("No data for the selected filters.")
    else:
        monthly_target = st.selectbox(
            "Pollutant for monthly distribution",
            options=selected_polls,
            format_func=lambda k: pollutant_opts.get(k, k),
            key="monthly_poll",
        )
        fig_monthly = plot_monthly_stats(
            filtered,
            pollutant=monthly_target,
            city_col="city",
            time_col=time_col,
        )
        st.plotly_chart(fig_monthly, use_container_width=True)


# ── Tab 3: Yearly Trend ─────────────────────────────────────

with tab3:
    if not selected_polls:
        st.info("Select at least one pollutant above.")
    elif filtered.empty:
        st.warning("No data for the selected filters.")
    else:
        yearly_target = st.selectbox(
            "Pollutant for yearly trend",
            options=selected_polls,
            format_func=lambda k: pollutant_opts.get(k, k),
            key="yearly_poll",
        )
        fig_yearly = plot_yearly_trend(
            filtered,
            pollutant=yearly_target,
            city_col="city",
            time_col=time_col,
        )
        st.plotly_chart(fig_yearly, use_container_width=True)


# ── Tab 4: Summary Statistics ───────────────────────────────

with tab4:
    if filtered.empty:
        st.warning("No data for the selected filters.")
    else:
        st.markdown("#### Per-city pollutant statistics")
        summary = compute_city_summary(filtered)
        if not summary.empty:
            st.dataframe(summary.round(3), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download summary statistics",
                data=summary.to_csv(index=False),
                file_name="city_pollution_summary.csv",
                mime="text/csv",
                key="hist_dl_summary",
            )

        st.markdown("#### Latest observed values per city")
        latest = get_latest_values(filtered)
        if not latest.empty:
            display_latest = latest[
                [c for c in ["city", time_col, "aqi", "pm25", "pm10", "no2", "so2", "o3", "co"]
                 if c in latest.columns]
            ].round(3)
            st.dataframe(display_latest, use_container_width=True, hide_index=True)

        st.markdown("#### Missing data report")
        with st.expander("View missing data counts", expanded=False):
            missing = compute_missing_report(filtered)
            st.dataframe(missing, use_container_width=True)
