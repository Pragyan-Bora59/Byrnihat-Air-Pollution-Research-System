"""
backend/chart_builder.py

Plotly figure factory for the Byrnihat Streamlit app.
All functions return plotly.graph_objects.Figure objects.
Display using st.plotly_chart(fig, use_container_width=True).

Reuses the graph data structures produced by ml/10_run_final_prediction_pipeline.py.
Never generates HTML or writes files.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Colour palette (consistent across all charts)
# ---------------------------------------------------------------------------

COLOUR = {
    "recorded":   "#60a5fa",   # blue
    "predicted":  "#f97316",   # orange
    "ml2_label":  "#a78bfa",   # violet
    "ml2_pred":   "#34d399",   # emerald
    "error":      "#f43f5e",   # rose
    "neutral":    "#94a3b8",   # slate
    "pm25":       "#fbbf24",
    "pm10":       "#fb923c",
    "so2":        "#a3e635",
    "co":         "#4ade80",
    "no2":        "#38bdf8",
    "o3":         "#c084fc",
    "nh3":        "#f472b6",
    "no":         "#fdba74",
    "nox":        "#67e8f9",
    "wind_u":     "#93c5fd",
    "wind_v":     "#86efac",
}

PLOTLY_TEMPLATE = "plotly_dark"


# ---------------------------------------------------------------------------
# ML-1 charts
# ---------------------------------------------------------------------------

def plot_ml1_actual_vs_predicted_line(graph_data, target="pm25"):
    """
    Time-series line chart: recorded vs ML-1 predicted for a given pollutant.

    graph_data: dict from load_graph_data() — ml1_recorded_vs_predicted[target]
    target: one of pm25, pm10, so2, co, no2, o3, nh3, no, nox, wind_u, wind_v
    """
    series = graph_data.get("ml1_recorded_vs_predicted", {}).get(target, [])

    if not series:
        fig = go.Figure()
        fig.add_annotation(text=f"No data available for {target}",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16, color="gray"))
        fig.update_layout(template=PLOTLY_TEMPLATE,
                          title=f"ML-1: {target.upper()} — No data")
        return fig

    df = pd.DataFrame(series)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["recorded"],
        name="Recorded",
        mode="lines",
        line=dict(color=COLOUR["recorded"], width=1.5),
        opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["predicted"],
        name="ML-1 Predicted",
        mode="lines",
        line=dict(color=COLOUR["predicted"], width=1.5, dash="dot"),
        opacity=0.9,
    ))

    fig.update_layout(
        title=f"ML-1: {target.upper()} — Recorded vs Predicted",
        xaxis_title="Datetime",
        yaxis_title=target.upper(),
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    return fig


def plot_ml1_scatter_actual_vs_predicted(df, target="pm25"):
    """
    Scatter plot: actual vs predicted (1:1 parity line) for ML-1 target.

    df: final predictions DataFrame (from load_final_predictions)
    """
    actual_col    = target
    predicted_col = f"ml1_predicted_{target}"

    if actual_col not in df.columns or predicted_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Columns for {target} not found",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color="gray"))
        fig.update_layout(template=PLOTLY_TEMPLATE,
                          title=f"ML-1 Scatter: {target.upper()}")
        return fig

    valid = df[[actual_col, predicted_col]].dropna()

    # 1:1 parity line
    all_vals = pd.concat([valid[actual_col], valid[predicted_col]])
    lo, hi = float(all_vals.min()), float(all_vals.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid[actual_col], y=valid[predicted_col],
        mode="markers",
        marker=dict(color=COLOUR["predicted"], size=4, opacity=0.5),
        name="Predictions",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(color=COLOUR["recorded"], dash="dash", width=1.5),
        name="Perfect prediction (1:1)",
    ))

    fig.update_layout(
        title=f"ML-1 Scatter: {target.upper()} Actual vs Predicted",
        xaxis_title=f"Actual {target.upper()}",
        yaxis_title=f"Predicted {target.upper()}",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def plot_ml1_residuals(df, target="pm25"):
    """
    Residual plot: (predicted - actual) vs actual for ML-1 target.
    """
    actual_col    = target
    predicted_col = f"ml1_predicted_{target}"

    if actual_col not in df.columns or predicted_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,
                          title=f"Residuals: {target.upper()} — Not available")
        return fig

    valid = df[[actual_col, predicted_col]].dropna().copy()
    valid["residual"] = valid[predicted_col] - valid[actual_col]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid[actual_col], y=valid["residual"],
        mode="markers",
        marker=dict(color=COLOUR["error"], size=4, opacity=0.5),
        name="Residual",
    ))
    fig.add_hline(y=0, line=dict(color="white", dash="dash", width=1))

    fig.update_layout(
        title=f"Residuals: {target.upper()} (Predicted − Actual)",
        xaxis_title=f"Actual {target.upper()}",
        yaxis_title="Residual",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def plot_ml1_metrics_bar(metrics_df):
    """
    Grouped bar chart comparing RMSE / MAE / R² across all ML-1 targets.

    metrics_df: DataFrame from load_ml1_metrics() or compute_metrics_for_all_targets()
    Expected columns: target, rmse, mae, r2
    """
    if metrics_df.empty:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE, title="ML-1 Metrics — No data")
        return fig

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["RMSE (lower = better)",
                                        "MAE (lower = better)",
                                        "R² (higher = better)"])

    for col_idx, (metric, colour) in enumerate(
        [("rmse", "#60a5fa"), ("mae", "#34d399"), ("r2", "#f97316")], start=1
    ):
        if metric not in metrics_df.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=metrics_df["target"],
                y=metrics_df[metric],
                marker_color=colour,
                name=metric.upper(),
                showlegend=False,
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(
        title="ML-1 Model Evaluation Metrics",
        template=PLOTLY_TEMPLATE,
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# ML-2 charts
# ---------------------------------------------------------------------------

def plot_ml2_plume_angle_comparison(graph_data):
    """
    Line chart: simulation-label plume angle vs ML-2 predicted plume angle over time.
    """
    series = (graph_data.get("ml2_simulation_label_vs_prediction", {})
              .get("plume_angle", []))

    if not series:
        fig = go.Figure()
        fig.add_annotation(text="No ML-2 plume angle data available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color="gray"))
        fig.update_layout(template=PLOTLY_TEMPLATE,
                          title="ML-2: Plume Angle — Simulation Label vs Predicted")
        return fig

    df = pd.DataFrame(series)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["simulation_label"],
        name="Simulation Label (ground truth)",
        mode="lines",
        line=dict(color=COLOUR["ml2_label"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["ml2_predicted"],
        name="ML-2 Predicted",
        mode="lines",
        line=dict(color=COLOUR["ml2_pred"], width=1.5, dash="dot"),
    ))

    fig.update_layout(
        title="ML-2: Plume Angle — Simulation Label vs Predicted",
        xaxis_title="Datetime",
        yaxis_title="Plume Angle (°)",
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    return fig


def plot_ml2_error_timeseries(graph_data):
    """
    Line chart: ML-2 angular error and vector error over time.
    """
    error_data = (graph_data.get("error_timeseries", {})
                  .get("plume_prediction_errors", []))

    if not error_data:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,
                          title="ML-2 Prediction Errors — No data")
        return fig

    df = pd.DataFrame(error_data)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Angular Error (°)", "Vector Error (U,V space)"])

    if "angular_error" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["angular_error"],
            name="Angular Error (°)", mode="lines",
            line=dict(color=COLOUR["error"], width=1.2),
        ), row=1, col=1)

    if "vector_error" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["vector_error"],
            name="Vector Error", mode="lines",
            line=dict(color=COLOUR["neutral"], width=1.2),
        ), row=2, col=1)

    fig.update_layout(
        title="ML-2: Prediction Error Over Time",
        template=PLOTLY_TEMPLATE,
        height=460,
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Feature importance charts
# ---------------------------------------------------------------------------

def plot_feature_importance(importance_df, top_n=20, title="Feature Importance"):
    """
    Horizontal bar chart for feature importance.

    importance_df: DataFrame with columns 'feature' and 'importance'
                   (and optionally 'target' for ML-1 multi-target importance)
    top_n: show top N features
    """
    if importance_df.empty or "feature" not in importance_df.columns:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE, title=title + " — No data")
        return fig

    if "importance" not in importance_df.columns:
        # Try to detect numeric column
        num_cols = importance_df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            importance_df = importance_df.copy()
            importance_df["importance"] = importance_df[num_cols[0]]
        else:
            fig = go.Figure()
            fig.update_layout(template=PLOTLY_TEMPLATE, title=title + " — No numeric column")
            return fig

    # Aggregate if multiple targets
    if "target" in importance_df.columns:
        agg = (importance_df.groupby("feature")["importance"]
               .mean().reset_index())
    else:
        agg = importance_df[["feature", "importance"]].copy()

    agg = agg.nlargest(top_n, "importance").sort_values("importance")

    fig = go.Figure(go.Bar(
        x=agg["importance"], y=agg["feature"],
        orientation="h",
        marker_color=COLOUR["predicted"],
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Mean Importance",
        yaxis_title="Feature",
        template=PLOTLY_TEMPLATE,
        height=max(300, top_n * 22),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Historical / time-series charts
# ---------------------------------------------------------------------------

def plot_pollution_timeseries(df, pollutants, city_col="city", time_col=None):
    """
    Multi-line time-series for one or more pollutants, coloured by city.

    df: combined AQICN DataFrame
    pollutants: list of pollutant column names to plot
    """
    # Detect time column
    if time_col is None:
        for candidate in ["api_time", "datetime"]:
            if candidate in df.columns:
                time_col = candidate
                break

    if time_col is None or df.empty:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,
                          title="Pollution Time Series — No data")
        return fig

    # One subplot per pollutant
    n = len(pollutants)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[p.upper() for p in pollutants])

    city_colours = px.colors.qualitative.Plotly
    cities = df[city_col].unique() if city_col in df.columns else ["All"]

    for row_idx, pollutant in enumerate(pollutants, start=1):
        if pollutant not in df.columns:
            continue

        if city_col in df.columns:
            for c_idx, city in enumerate(cities):
                city_df = df[df[city_col] == city].sort_values(time_col)
                fig.add_trace(go.Scatter(
                    x=city_df[time_col], y=city_df[pollutant],
                    name=city,
                    mode="lines+markers",
                    marker=dict(size=4),
                    line=dict(color=city_colours[c_idx % len(city_colours)], width=1.5),
                    showlegend=(row_idx == 1),
                ), row=row_idx, col=1)
        else:
            city_df = df.sort_values(time_col)
            fig.add_trace(go.Scatter(
                x=city_df[time_col], y=city_df[pollutant],
                name=pollutant,
                mode="lines+markers",
                marker=dict(size=4),
            ), row=row_idx, col=1)

    fig.update_layout(
        title="Pollution Levels Over Time",
        template=PLOTLY_TEMPLATE,
        height=max(350, 220 * n),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
    )
    return fig


def plot_monthly_stats(df, pollutant="pm25", city_col="city", time_col=None):
    """
    Box plot of monthly pollutant distribution, grouped by city.
    """
    if time_col is None:
        for candidate in ["api_time", "datetime"]:
            if candidate in df.columns:
                time_col = candidate
                break

    if time_col is None or pollutant not in df.columns or df.empty:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,
                          title=f"Monthly Stats: {pollutant.upper()} — No data")
        return fig

    df = df.copy()
    df["_month"] = pd.to_datetime(df[time_col], errors="coerce").dt.to_period("M").astype(str)

    fig = px.box(
        df, x="_month", y=pollutant,
        color=city_col if city_col in df.columns else None,
        title=f"Monthly Distribution: {pollutant.upper()}",
        template=PLOTLY_TEMPLATE,
        labels={"_month": "Month", pollutant: pollutant.upper()},
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_yearly_trend(df, pollutant="pm25", city_col="city", time_col=None):
    """
    Yearly mean trend line chart.
    """
    if time_col is None:
        for candidate in ["api_time", "datetime"]:
            if candidate in df.columns:
                time_col = candidate
                break

    if time_col is None or pollutant not in df.columns or df.empty:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,
                          title=f"Yearly Trend: {pollutant.upper()} — No data")
        return fig

    df = df.copy()
    df["_year"] = pd.to_datetime(df[time_col], errors="coerce").dt.year
    df = df.dropna(subset=["_year"])

    group_cols = ["_year", city_col] if city_col in df.columns else ["_year"]
    agg = df.groupby(group_cols)[pollutant].mean().reset_index()

    if city_col in df.columns:
        fig = px.line(agg, x="_year", y=pollutant, color=city_col,
                      markers=True,
                      title=f"Yearly Mean: {pollutant.upper()}",
                      template=PLOTLY_TEMPLATE,
                      labels={"_year": "Year", pollutant: f"Mean {pollutant.upper()}"})
    else:
        fig = px.line(agg, x="_year", y=pollutant, markers=True,
                      title=f"Yearly Mean: {pollutant.upper()}",
                      template=PLOTLY_TEMPLATE,
                      labels={"_year": "Year", pollutant: f"Mean {pollutant.upper()}"})

    return fig


# ---------------------------------------------------------------------------
# Simulation charts
# ---------------------------------------------------------------------------

def plot_diffusion_heatmap(grid, title="PM10 Diffusion Heatmap"):
    """
    Plotly heatmap of the 2D pollution diffusion grid.

    grid: 2D numpy array (rows × cols)
    """
    fig = go.Figure(go.Heatmap(
        z=grid,
        colorscale="YlOrRd",
        colorbar=dict(title="PM10 Concentration"),
        zmin=0,
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Grid Column",
        yaxis_title="Grid Row",
        template=PLOTLY_TEMPLATE,
        height=500,
    )
    return fig


def plot_entropy_vs_time(entropy_values, title="Entropy Spread of PM10 Over Time"):
    """
    Line chart of Shannon entropy at each simulation time step.

    entropy_values: list or array of entropy values
    """
    fig = go.Figure(go.Scatter(
        x=list(range(len(entropy_values))),
        y=entropy_values,
        mode="lines",
        line=dict(color=COLOUR["ml2_pred"], width=2),
        name="Entropy",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="Entropy",
        template=PLOTLY_TEMPLATE,
        height=350,
    )
    return fig


def plot_wind_rose(df, speed_col="wind_speed", dir_col="wind_direction", title="Wind Rose"):
    """
    Wind rose (polar bar) chart.

    df: DataFrame with wind_speed and wind_direction columns
    """
    if speed_col not in df.columns or dir_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE, title=title + " — No data")
        return fig

    valid = df[[speed_col, dir_col]].dropna()

    if valid.empty:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE, title=title + " — No valid data")
        return fig

    # Bin directions into 16 compass sectors
    bins = np.arange(0, 361, 22.5)
    labels = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

    valid = valid.copy()
    valid["sector"] = pd.cut(
        valid[dir_col] % 360,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    rose = valid.groupby("sector")[speed_col].mean().reset_index()

    fig = go.Figure(go.Barpolar(
        r=rose[speed_col],
        theta=rose["sector"].astype(str),
        marker_color=COLOUR["recorded"],
        opacity=0.8,
    ))

    fig.update_layout(
        title=title,
        polar=dict(bgcolor="#1e293b"),
        template=PLOTLY_TEMPLATE,
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Plume animation frame
# ---------------------------------------------------------------------------

def plot_plume_frame(frames_df, record_id, frame_index, source_lat, source_lon,
                     mapbox_token=None):
    """
    Single Plotly Scattermapbox frame for one plume animation step.

    frames_df   : plume frames DataFrame from load_plume_frames()
    record_id   : integer record index
    frame_index : 0–29 animation frame
    source_lat  : source latitude (for marker)
    source_lon  : source longitude (for marker)

    Returns a Plotly Figure.
    """
    frame = frames_df[
        (frames_df["record_id"] == record_id) &
        (frames_df["frame_index"] == frame_index)
    ].copy()

    # Particle scatter
    fig = go.Figure()

    if not frame.empty:
        fig.add_trace(go.Scattermapbox(
            lat=frame["lat"],
            lon=frame["lon"],
            mode="markers",
            marker=dict(
                size=frame["size"] * 2.5,
                color=frame["opacity"].apply(
                    lambda o: f"rgba(255,165,0,{min(o, 1.0):.2f})"
                ),
                sizemode="area",
            ),
            name="Plume particles",
            hoverinfo="skip",
        ))

    # Source marker
    fig.add_trace(go.Scattermapbox(
        lat=[source_lat],
        lon=[source_lon],
        mode="markers",
        marker=dict(size=14, color="#ef4444", symbol="circle"),
        name="Source",
        hovertemplate="Industrial Source<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
    ))

    # Center map on source
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=source_lat, lon=source_lon),
            zoom=10,
            accesstoken=mapbox_token,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=520,
        template=PLOTLY_TEMPLATE,
        title=f"Plume Animation — Record {record_id} | Frame {frame_index + 1}/30",
        showlegend=True,
    )
    return fig


def plot_industry_sources_map(sources_df):
    """
    Interactive map of industrial sources with pollutant weight colour coding.

    sources_df: DataFrame from load_industry_sources()
    Returns a Plotly Figure (Scattermapbox).
    """
    if sources_df.empty:
        fig = go.Figure()
        fig.update_layout(template=PLOTLY_TEMPLATE,
                          title="Industry Sources — No data")
        return fig

    # Scale marker size by pm10_weight (clamped)
    if "pm10_weight" in sources_df.columns:
        sizes = (sources_df["pm10_weight"].fillna(1.0)
                 .clip(lower=1, upper=100)
                 .apply(lambda w: 6 + w * 0.3))
    else:
        sizes = 10

    hover_cols = ["industry_name", "industry_type", "pm10_weight",
                  "pm25_weight", "so2_weight", "nox_weight",
                  "elevation_m", "terrain_class"]

    hover_text = []
    for _, row in sources_df.iterrows():
        parts = []
        for col in hover_cols:
            if col in sources_df.columns and pd.notna(row.get(col)):
                parts.append(f"<b>{col}</b>: {row[col]}")
        hover_text.append("<br>".join(parts))

    fig = go.Figure(go.Scattermapbox(
        lat=sources_df["latitude"],
        lon=sources_df["longitude"],
        mode="markers",
        marker=dict(
            size=sizes,
            color="#ef4444",
            opacity=0.75,
        ),
        text=hover_text,
        hoverinfo="text",
        name="Industrial Sources",
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=26.065, lon=91.875),
            zoom=10,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=520,
        title="Byrnihat Industrial Emission Sources",
        template=PLOTLY_TEMPLATE,
    )
    return fig
