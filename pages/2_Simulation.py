"""
pages/2_Simulation.py — Simulation & Maps

Displays:
  - Plume animation (particle frames from ML-2 predictions)
  - PM10 diffusion grid simulation (Laplacian)
  - Wind vector map
  - Terrain / industry source map

Reuses: simulation_engine.py, map_builder.py, chart_builder.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data_loader       import load_plume_frames, load_industry_sources, load_final_predictions
from backend.map_builder       import (
    build_plume_frame_map,
    build_industry_source_map,
    build_wind_vector_map,
    build_terrain_pydeck,
    build_terrain_plotly_fallback,
)
from backend.chart_builder     import plot_diffusion_heatmap, plot_entropy_vs_time
from backend.simulation_engine import run_diffusion_simulation, get_record_summary


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Simulation — Byrnihat",
    page_icon="🌫️",
    layout="wide",
)

CSS = Path(__file__).parents[1] / "assets" / "style.css"
if CSS.exists():
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

st.markdown("## 🌫️ Simulation & Maps")
st.markdown(
    "<div style='color:#94a3b8; margin-bottom:1.5rem'>"
    "Interactive plume animation, PM10 diffusion grid, wind vectors and terrain overview."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🌫️ Plume Animation",
    "🔥 Diffusion Grid",
    "💨 Wind Vectors",
    "🗺️ Terrain & Sources",
])


# ============================================================
# TAB 1 — Plume Animation
# ============================================================

with tab1:
    st.markdown("### ML-2 Predicted Plume Animation")
    st.markdown(
        "<div style='color:#94a3b8; font-size:0.9rem; margin-bottom:1rem'>"
        "Particle simulation driven by ML-2 predicted plume U/V vectors. "
        "Select a record and use the frame slider to step through the animation."
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        frames_df = load_plume_frames()
        pred_df   = load_final_predictions()
        plume_ok  = True
    except FileNotFoundError as e:
        st.error(f"Plume data not found: {e}\n\nRun the prediction pipeline first.")
        plume_ok = False

    if plume_ok and not frames_df.empty and not pred_df.empty:
        # Controls row
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])

        record_ids = sorted(frames_df["record_id"].unique())
        total_records = len(record_ids)

        with ctrl_col1:
            # Let user pick by datetime if available
            if "datetime" in pred_df.columns:
                dt_series = pred_df["datetime"].astype(str)
                dt_options = dt_series.tolist()
                sel_idx = st.selectbox(
                    "Select record (datetime)",
                    options=range(min(len(dt_options), len(record_ids))),
                    format_func=lambda i: f"[{i}] {dt_options[i][:19]}",
                    key="plume_record_sel",
                )
            else:
                sel_idx = st.selectbox(
                    "Select record ID",
                    options=record_ids[:200],
                    key="plume_record_sel_id",
                )
            record_id = record_ids[sel_idx] if sel_idx < len(record_ids) else record_ids[0]

        with ctrl_col2:
            frame_index = st.slider(
                "Animation frame (0 = near source, 29 = dispersed)",
                min_value=0, max_value=29, step=1,
                key="plume_frame_slider",
            )

        with ctrl_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            play_animation = st.button("▶ Play Animation", key="plume_play")

        # Determine source coordinates from predictions
        source_lat = 26.065
        source_lon = 91.875

        if record_id < len(pred_df):
            row = pred_df.iloc[record_id]
            for lat_col in ["source_lat", "latitude"]:
                if lat_col in pred_df.columns and not pd.isna(row.get(lat_col)):
                    source_lat = float(row[lat_col])
                    break
            for lon_col in ["source_lon", "longitude"]:
                if lon_col in pred_df.columns and not pd.isna(row.get(lon_col)):
                    source_lon = float(row[lon_col])
                    break

        # Map
        map_placeholder = st.empty()
        
        if play_animation:
            import time
            for f in range(30):
                fig_plume = build_plume_frame_map(
                    frames_df, record_id, f, source_lat, source_lon
                )
                map_placeholder.plotly_chart(fig_plume, use_container_width=True)
                time.sleep(0.1)
            # Animation complete, we trigger a rerun to reset the play button state
            st.rerun()
        else:
            fig_plume = build_plume_frame_map(
                frames_df, record_id, frame_index, source_lat, source_lon
            )
            map_placeholder.plotly_chart(fig_plume, use_container_width=True)

        # Info panel below the map
        st.markdown("---")
        summary = get_record_summary(pred_df, record_id)
        if summary:
            i1, i2, i3 = st.columns(3)
            with i1:
                st.markdown("**📍 Source**")
                st.write(f"Name: {summary.get('source_name', summary.get('industry_name', 'N/A'))}")
                st.write(f"Lat: {summary.get('source_lat', 'N/A')}")
                st.write(f"Lon: {summary.get('source_lon', 'N/A')}")
            with i2:
                st.markdown("**🌫️ ML-2 Plume**")
                st.write(f"U: {summary.get('ml2_predicted_plume_u', 'N/A')}")
                st.write(f"V: {summary.get('ml2_predicted_plume_v', 'N/A')}")
                st.write(f"Angle: {summary.get('ml2_predicted_plume_angle', 'N/A')} °")
                st.write(f"Strength: {summary.get('ml2_predicted_plume_strength', 'N/A')}")
            with i3:
                st.markdown("**💨 ML-1 Pollution**")
                st.write(f"PM2.5: {summary.get('ml1_predicted_pm25', summary.get('pm25', 'N/A'))}")
                st.write(f"PM10:  {summary.get('ml1_predicted_pm10', summary.get('pm10', 'N/A'))}")
                st.write(f"Wind Speed: {summary.get('wind_speed', 'N/A')}")
                st.write(f"Rainfall: {summary.get('rainfall', 'N/A')}")

        # Record navigation
        st.caption(f"Record {record_id} of {total_records - 1} available | "
                   f"Frame {frame_index + 1} of 30")

    elif plume_ok:
        st.warning("Plume frames file is empty. "
                   "Re-run `ml/10_run_final_prediction_pipeline.py`.")


# ============================================================
# TAB 2 — Diffusion Grid
# ============================================================

with tab2:
    st.markdown("### PM10 Diffusion Grid Simulation")
    st.markdown(
        "<div style='color:#94a3b8; font-size:0.9rem; margin-bottom:1rem'>"
        "2D Laplacian diffusion simulation seeded from industrial emission sources. "
        "Adjust parameters and run to see how PM10 disperses across the region."
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        sources_df = load_industry_sources()
        sources_ok = not sources_df.empty
    except Exception as e:
        sources_ok = False
        st.error(f"Could not load industry sources: {e}")
        sources_df = pd.DataFrame()

    # Simulation parameters
    param_col1, param_col2, param_col3, param_col4 = st.columns(4)
    with param_col1:
        time_steps = st.slider("Time Steps", 10, 300, 150, 10,
                               help="Number of simulation steps (default: 150)")
    with param_col2:
        diffusion_rate = st.slider("Diffusion Rate", 0.01, 0.40, 0.15, 0.01,
                                   help="Laplacian diffusion coefficient")
    with param_col3:
        decay_rate = st.slider("Decay Rate", 0.900, 0.999, 0.995, 0.001,
                               help="Per-step concentration decay multiplier")
    with param_col4:
        grid_size = st.selectbox("Grid Size", [51, 101, 151], index=1,
                                 help="Simulation grid resolution")

    run_sim = st.button("🚀 Run Simulation", key="run_diffusion")

    if sources_ok and run_sim:
        with st.spinner(f"Running {time_steps}-step diffusion simulation…"):
            try:
                grid, entropy_list, bounds, source_cells = run_diffusion_simulation(
                    sources_df,
                    time_steps=time_steps,
                    diffusion_rate=diffusion_rate,
                    decay_rate=decay_rate,
                    grid_size=grid_size,
                )
                st.session_state["sim_grid"]    = grid
                st.session_state["entropy_list"] = entropy_list
                st.session_state["source_cells"] = source_cells
                st.success(f"Simulation complete: {time_steps} steps, "
                           f"grid {grid_size}×{grid_size}")
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    elif sources_ok and not run_sim:
        st.info("Adjust parameters above and click **Run Simulation** to start.")

    if "sim_grid" in st.session_state:
        grid_data    = st.session_state["sim_grid"]
        entropy_data = st.session_state["entropy_list"]

        h1, h2 = st.columns([2, 1])
        with h1:
            fig_heat = plot_diffusion_heatmap(grid_data,
                                              title="PM10 Diffusion — Final State")
            st.plotly_chart(fig_heat, use_container_width=True)
        with h2:
            fig_ent = plot_entropy_vs_time(entropy_data)
            st.plotly_chart(fig_ent, use_container_width=True)

            # Summary stats
            st.metric("Peak PM10 Concentration", f"{grid_data.max():.2f}")
            st.metric("Final Entropy", f"{entropy_data[-1]:.4f}")
            st.metric("Grid Size", f"{grid_data.shape[0]}×{grid_data.shape[1]}")

    if not sources_ok:
        st.warning("Industry source data not found. "
                   "Run `mapping/interactive_terrain_overlay_map_v2.py` or "
                   "check `data/` for the source CSV.")


# ============================================================
# TAB 3 — Wind Vectors
# ============================================================

with tab3:
    st.markdown("### Wind Vector Map")
    st.markdown(
        "<div style='color:#94a3b8; font-size:0.9rem; margin-bottom:1rem'>"
        "ML-1 predicted wind vectors (U, V components) displayed as directional "
        "arrows on the map. Each vector originates from an industrial source location."
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        pred_wind_df = load_final_predictions()
        wind_ok = True
    except FileNotFoundError:
        wind_ok = False
        st.error("Prediction output not found.")

    if wind_ok and not pred_wind_df.empty:
        # Date filter
        if "datetime" in pred_wind_df.columns:
            pred_wind_df["datetime"] = pd.to_datetime(pred_wind_df["datetime"],
                                                       errors="coerce")
            min_dt = pred_wind_df["datetime"].min()
            max_dt = pred_wind_df["datetime"].max()

            wc1, wc2 = st.columns(2)
            with wc1:
                wind_date = st.date_input(
                    "Select date",
                    value=max_dt.date() if pd.notna(max_dt) else None,
                    min_value=min_dt.date() if pd.notna(min_dt) else None,
                    max_value=max_dt.date() if pd.notna(max_dt) else None,
                    key="wind_date",
                )
            with wc2:
                n_vectors = st.slider("Max vectors to display", 50, 500, 200, 50,
                                      key="wind_n_vec")

            date_mask = pred_wind_df["datetime"].dt.date == pd.Timestamp(wind_date).date()
            wind_subset = pred_wind_df[date_mask] if date_mask.any() else pred_wind_df.tail(500)
        else:
            wind_subset = pred_wind_df.tail(500)
            n_vectors   = 200

        fig_wind = build_wind_vector_map(wind_subset, sample_n=n_vectors)
        st.plotly_chart(fig_wind, use_container_width=True)

        # Wind stats
        u_col = "ml1_predicted_wind_u" if "ml1_predicted_wind_u" in wind_subset.columns else "wind_u"
        v_col = "ml1_predicted_wind_v" if "ml1_predicted_wind_v" in wind_subset.columns else "wind_v"

        if u_col in wind_subset.columns and v_col in wind_subset.columns:
            speeds = np.sqrt(
                wind_subset[u_col].dropna() ** 2 +
                wind_subset[v_col].dropna() ** 2
            )
            ws1, ws2, ws3 = st.columns(3)
            with ws1:
                st.metric("Mean Wind Speed", f"{speeds.mean():.2f} m/s")
            with ws2:
                st.metric("Max Wind Speed", f"{speeds.max():.2f} m/s")
            with ws3:
                st.metric("Vectors Shown", f"{min(n_vectors, len(wind_subset)):,}")
    elif wind_ok:
        st.warning("No prediction data available.")


# ============================================================
# TAB 4 — Terrain & Sources
# ============================================================

with tab4:
    st.markdown("### Terrain Overview & Industrial Sources")
    st.markdown(
        "<div style='color:#94a3b8; font-size:0.9rem; margin-bottom:1rem'>"
        "Interactive map of Byrnihat region showing industrial emission sources. "
        "Terrain hillshade overlay uses the pre-generated DEM PNG."
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        sources_t = load_industry_sources()
        sources_t_ok = not sources_t.empty
    except Exception as e:
        sources_t_ok = False
        sources_t = pd.DataFrame()
        st.error(f"Could not load sources: {e}")

    view_mode = st.radio(
        "Map view",
        ["Industry Sources (Plotly)", "Terrain + Sources (PyDeck)"],
        horizontal=True,
        key="terrain_view",
    )

    if "Terrain" in view_mode:
        # Try PyDeck first
        deck = build_terrain_pydeck(sources_t) if sources_t_ok else None
        if deck is not None:
            st.pydeck_chart(deck, use_container_width=True)
            terrain_png = Path(__file__).parents[1] / "outputs" / "maps" / "final_terrain_hillshade_overlay.png"
            if terrain_png.exists():
                st.success("✅ DEM hillshade overlay loaded from "
                           "`outputs/maps/final_terrain_hillshade_overlay.png`")
            else:
                st.warning("DEM hillshade PNG not found. "
                           "Run `mapping/interactive_terrain_overlay_map_v2.py` to generate it.")
        else:
            st.info("PyDeck not available — falling back to Plotly map.")
            fig_t = build_terrain_plotly_fallback(sources_t)
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        fig_src = build_industry_source_map(sources_t)
        st.plotly_chart(fig_src, use_container_width=True)

    # Source data table
    if sources_t_ok:
        with st.expander("📋 Industry Source Data Table", expanded=False):
            display_cols = [c for c in [
                "industry_name", "industry_type", "latitude", "longitude",
                "pm10_weight", "pm25_weight", "so2_weight", "nox_weight",
                "elevation_m", "terrain_class", "confidence",
            ] if c in sources_t.columns]
            st.dataframe(
                sources_t[display_cols].round(4),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"{len(sources_t)} active sources loaded")
