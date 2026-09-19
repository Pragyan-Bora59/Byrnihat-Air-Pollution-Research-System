"""
backend/map_builder.py

Replaces mapping/interactive_terrain_overlay_map_v2.py HTML generation with
Python-native map objects (Plotly Scattermapbox, PyDeck).
The original Folium/Leaflet HTML is NOT used here.
All functions return Plotly Figures or PyDeck Deck objects — never HTML strings.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPS_DIR     = PROJECT_ROOT / "outputs" / "maps"

# Study area constants (from interactive_terrain_overlay_map_v2.py)
MAP_CENTER   = {"lat": 26.04, "lon": 91.83}
STUDY_BOUNDS = {"lat": [25.50, 26.22], "lon": [91.62, 92.00]}
ZOOM_DEFAULT = 10

TERRAIN_PNG  = MAPS_DIR / "final_terrain_hillshade_overlay.png"


# ---------------------------------------------------------------------------
# Industry source map (replaces add_industry_sources Folium layer)
# ---------------------------------------------------------------------------

def build_industry_source_map(sources_df, zoom=10):
    """
    Interactive Plotly Scattermapbox map of Byrnihat industrial emission sources.
    Replaces the Folium industry layer from interactive_terrain_overlay_map_v2.py.

    sources_df : DataFrame from load_industry_sources()
    Returns    : plotly.graph_objects.Figure
    """
    fig = go.Figure()

    if sources_df.empty:
        fig.update_layout(
            mapbox=dict(style="carto-darkmatter",
                        center=MAP_CENTER, zoom=zoom),
            height=540, title="No industry source data found",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    # Marker size scaled by pm10_weight
    weight_col = "pm10_weight" if "pm10_weight" in sources_df.columns else None
    if weight_col:
        sizes = (sources_df[weight_col].fillna(1.0)
                 .clip(lower=1, upper=200)
                 .apply(lambda w: 8 + w * 0.25))
    else:
        sizes = pd.Series([10] * len(sources_df))

    # Build hover text matching original Folium popup content
    hover_lines = []
    info_cols = [
        ("industry_name",           "Name"),
        ("industry_type",           "Type"),
        ("source_type",             "Source type"),
        ("pm10_weight",             "PM10 weight"),
        ("pm25_weight",             "PM2.5 weight"),
        ("so2_weight",              "SO2 weight"),
        ("nox_weight",              "NOx weight"),
        ("elevation_m",             "Elevation (m)"),
        ("terrain_class",           "Terrain class"),
        ("terrain_exposure_factor", "Terrain exposure"),
        ("adjusted_pm10_weight",    "Adj. PM10 weight"),
        ("confidence",              "Confidence"),
        ("coordinate_status",       "Coord. status"),
    ]
    for _, row in sources_df.iterrows():
        parts = []
        for col, label in info_cols:
            if col in sources_df.columns and pd.notna(row.get(col)):
                parts.append(f"<b>{label}:</b> {row[col]}")
        hover_lines.append("<br>".join(parts))

    fig.add_trace(go.Scattermapbox(
        lat=sources_df["latitude"],
        lon=sources_df["longitude"],
        mode="markers",
        marker=dict(size=sizes, color="#ef4444", opacity=0.80),
        text=hover_lines,
        hoverinfo="text",
        name="Industrial Sources",
    ))

    # Major places (mirrors add_major_places from original script)
    places = [
        {"name": "Byrnihat",              "lat": 26.065,  "lon": 91.875,  "desc": "Main industrial study area"},
        {"name": "Guwahati",              "lat": 26.1445, "lon": 91.7362, "desc": "Major nearby urban area (Assam)"},
        {"name": "Sonapur / Jorabat",     "lat": 26.105,  "lon": 91.83,   "desc": "Transport corridor"},
        {"name": "Shillong (reference)",  "lat": 25.5788, "lon": 91.8933, "desc": "High-elevation reference city"},
    ]
    fig.add_trace(go.Scattermapbox(
        lat=[p["lat"] for p in places],
        lon=[p["lon"] for p in places],
        mode="markers+text",
        marker=dict(size=12, color="#60a5fa", symbol="circle"),
        text=[p["name"] for p in places],
        textposition="top right",
        hovertext=[f"<b>{p['name']}</b><br>{p['desc']}" for p in places],
        hoverinfo="text",
        name="Major Places",
        textfont=dict(color="white", size=11),
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=MAP_CENTER,
            zoom=zoom,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=560,
        title="Byrnihat Region — Industrial Sources & Major Places",
        legend=dict(
            orientation="v", x=0.01, y=0.99,
            bgcolor="rgba(30,41,59,0.8)",
            font=dict(color="white"),
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
    )
    return fig


# ---------------------------------------------------------------------------
# Wind vector map
# ---------------------------------------------------------------------------

def build_wind_vector_map(df, sample_n=200, zoom=10):
    """
    Plotly map with wind vectors shown as arrows (quiver-style using Scattermapbox lines).

    df       : predictions DataFrame with source_lat/source_lon, wind_u, wind_v columns
    sample_n : max number of vectors to draw (for performance)
    Returns  : plotly.graph_objects.Figure
    """
    lat_col = "source_lat" if "source_lat" in df.columns else ("station_latitude" if "station_latitude" in df.columns else "latitude")
    lon_col = "source_lon" if "source_lon" in df.columns else ("station_longitude" if "station_longitude" in df.columns else "longitude")

    if lat_col not in df.columns or lon_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Wind Vectors — Missing lat/lon data",
                          height=500,
                          paper_bgcolor="#0f172a")
        return fig

    required = [lat_col, lon_col]
    wind_u = "ml1_predicted_wind_u" if "ml1_predicted_wind_u" in df.columns else "wind_u"
    wind_v = "ml1_predicted_wind_v" if "ml1_predicted_wind_v" in df.columns else "wind_v"

    valid = df[required + [wind_u, wind_v]].dropna()
    if valid.empty:
        fig = go.Figure()
        fig.update_layout(title="Wind Vectors — No valid data",
                          height=500,
                          paper_bgcolor="#0f172a")
        return fig

    # Sample for rendering performance
    if len(valid) > sample_n:
        valid = valid.sample(sample_n, random_state=42)

    scale = 0.03  # degrees per unit wind speed

    lats, lons, texts = [], [], []
    for _, row in valid.iterrows():
        lat0 = float(row[lat_col])
        lon0 = float(row[lon_col])
        u    = float(row[wind_u])
        v    = float(row[wind_v])

        lat1 = lat0 + v * scale
        lon1 = lon0 + u * scale
        speed = round(np.sqrt(u**2 + v**2), 2)

        lats += [lat0, lat1, None]
        lons += [lon0, lon1, None]
        texts += [f"Speed: {speed} m/s", "", ""]

    fig = go.Figure(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="lines",
        line=dict(width=1.5, color="#38bdf8"),
        hoverinfo="skip",
        name="Wind vectors",
    ))

    # Source origin points
    fig.add_trace(go.Scattermapbox(
        lat=valid[lat_col],
        lon=valid[lon_col],
        mode="markers",
        marker=dict(size=5, color="#f97316"),
        hovertext=texts[::3],
        hoverinfo="text",
        name="Source points",
    ))

    fig.update_layout(
        mapbox=dict(style="carto-darkmatter", center=MAP_CENTER, zoom=zoom),
        margin=dict(l=0, r=0, t=40, b=0),
        height=540,
        title="Wind Vectors (ML-1 Predicted)",
        paper_bgcolor="#0f172a",
        legend=dict(bgcolor="rgba(30,41,59,0.8)", font=dict(color="white")),
    )
    return fig


# ---------------------------------------------------------------------------
# Plume animation map (single frame)
# ---------------------------------------------------------------------------

def build_plume_frame_map(frames_df, record_id, frame_index,
                           source_lat, source_lon, zoom=11):
    """
    Single Plotly Scattermapbox frame for the plume animation.
    Replaces the Leaflet particle animation from the HTML dashboard.

    frames_df   : DataFrame from load_plume_frames()
    record_id   : integer record index
    frame_index : 0–29 animation frame
    source_lat/lon: emission source coordinates
    Returns     : plotly.graph_objects.Figure
    """
    frame = frames_df[
        (frames_df["record_id"] == record_id) &
        (frames_df["frame_index"] == frame_index)
    ].copy()

    fig = go.Figure()

    if not frame.empty:
        # Particle colours from orange→red based on opacity
        colours = [
            f"rgba(255, {max(100, int(165 - o * 100))}, 0, {min(o, 1.0):.2f})"
            for o in frame["opacity"].tolist()
        ]
        fig.add_trace(go.Scattermapbox(
            lat=frame["lat"],
            lon=frame["lon"],
            mode="markers",
            marker=dict(
                size=frame["size"].clip(upper=20) * 2,
                color=colours,
                sizemode="area",
            ),
            hoverinfo="skip",
            name="Plume particles",
        ))

        # Show plume centre path
        if "center_lat" in frame.columns and "center_lon" in frame.columns:
            center_row = frame.iloc[0]
            fig.add_trace(go.Scattermapbox(
                lat=[source_lat, center_row["center_lat"]],
                lon=[source_lon, center_row["center_lon"]],
                mode="lines",
                line=dict(color="#f97316", width=2),
                name="Plume centre path",
                hoverinfo="skip",
            ))

    # Emission source marker
    fig.add_trace(go.Scattermapbox(
        lat=[source_lat],
        lon=[source_lon],
        mode="markers",
        marker=dict(size=16, color="#ef4444", symbol="circle"),
        name="Emission source",
        hovertemplate=(f"<b>Emission Source</b><br>"
                       f"Lat: {source_lat:.4f}<br>"
                       f"Lon: {source_lon:.4f}<extra></extra>"),
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=source_lat, lon=source_lon),
            zoom=zoom,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=540,
        title=(f"Plume Animation — Record {record_id} | "
               f"Frame {frame_index + 1} / {30}"),
        legend=dict(
            orientation="h", x=0.01, y=0.01,
            bgcolor="rgba(30,41,59,0.85)",
            font=dict(color="white", size=11),
        ),
        paper_bgcolor="#0f172a",
    )
    return fig


# ---------------------------------------------------------------------------
# Terrain map via PyDeck (replaces Folium DEM overlay)
# ---------------------------------------------------------------------------

def build_terrain_pydeck(sources_df):
    """
    Build a PyDeck deck object with:
      - Terrain hillshade BitmapLayer (if PNG exists)
      - Industry sources ScatterplotLayer

    Returns a pydeck.Deck object, or None if pydeck is unavailable.
    """
    try:
        import pydeck as pdk
    except ImportError:
        return None

    layers = []

    # Industry source layer
    if not sources_df.empty:
        source_data = sources_df[["latitude", "longitude"]].copy()
        source_data = source_data.rename(columns={"latitude": "lat", "longitude": "lon"})
        source_data["radius"] = 200

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=source_data,
            get_position=["lon", "lat"],
            get_radius="radius",
            get_fill_color=[239, 68, 68, 180],
            pickable=True,
        ))

    # Terrain hillshade BitmapLayer
    if TERRAIN_PNG.exists():
        layers.append(pdk.Layer(
            "BitmapLayer",
            data=None,
            image=str(TERRAIN_PNG),
            bounds=[91.62, 25.50, 92.00, 26.22],  # [west, south, east, north]
            opacity=0.45,
        ))

    view_state = pdk.ViewState(
        latitude=MAP_CENTER["lat"],
        longitude=MAP_CENTER["lon"],
        zoom=9,
        pitch=0,
        bearing=0,
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="dark",
        tooltip={"text": "Industrial source"},
    )
    return deck


def build_terrain_plotly_fallback(sources_df):
    """
    Fallback terrain map using Plotly Scattermapbox with OpenStreetMap tile
    if PyDeck is unavailable or terrain PNG is missing.
    """
    return build_industry_source_map(sources_df)


# ---------------------------------------------------------------------------
# Study area overview map
# ---------------------------------------------------------------------------

def build_study_area_map(zoom=9):
    """
    Static overview map of the study area bounding box.
    Shows the Byrnihat region and major reference cities.
    Returns a Plotly Figure.
    """
    places = [
        {"name": "Byrnihat",             "lat": 26.065,  "lon": 91.875,  "colour": "#ef4444"},
        {"name": "Guwahati",             "lat": 26.1445, "lon": 91.7362, "colour": "#60a5fa"},
        {"name": "Sonapur / Jorabat",    "lat": 26.105,  "lon": 91.830,  "colour": "#34d399"},
        {"name": "Shillong (reference)", "lat": 25.5788, "lon": 91.8933, "colour": "#f59e0b"},
    ]

    fig = go.Figure()
    for p in places:
        fig.add_trace(go.Scattermapbox(
            lat=[p["lat"]], lon=[p["lon"]],
            mode="markers+text",
            marker=dict(size=14, color=p["colour"]),
            text=[p["name"]],
            textposition="top right",
            textfont=dict(color="white", size=12),
            name=p["name"],
            hovertemplate=f"<b>{p['name']}</b><br>Lat: {p['lat']}<br>Lon: {p['lon']}<extra></extra>",
        ))

    fig.update_layout(
        mapbox=dict(style="carto-darkmatter", center=MAP_CENTER, zoom=zoom),
        margin=dict(l=0, r=0, t=40, b=0),
        height=480,
        title="Study Area — Byrnihat Region, Assam–Meghalaya Border",
        paper_bgcolor="#0f172a",
        legend=dict(bgcolor="rgba(30,41,59,0.8)", font=dict(color="white")),
    )
    return fig
