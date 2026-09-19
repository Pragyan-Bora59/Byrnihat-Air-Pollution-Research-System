"""
backend/simulation_engine.py

Wraps simulation/pollution_grid_v2.py logic into importable, parameterizable functions.
The original script is NOT modified. Functions return numpy arrays and DataFrames — never HTML.

Also wraps the particle frame builder from ml/10_run_final_prediction_pipeline.py
for use in the Streamlit plume animation page.
"""

import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Simulation constants (mirrors pollution_grid_v2.py defaults)
# ---------------------------------------------------------------------------

DEFAULT_GRID_SIZE      = 101
DEFAULT_TIME_STEPS     = 150
DEFAULT_DIFFUSION_RATE = 0.15
DEFAULT_DECAY_RATE     = 0.995


# ---------------------------------------------------------------------------
# Grid helpers (from pollution_grid_v2.py)
# ---------------------------------------------------------------------------

def build_grid_bounds(sources_df, padding_fraction=0.10):
    """
    Compute lat/lon bounding box for the simulation grid.
    Returns (min_lat, max_lat, min_lon, max_lon).
    """
    min_lat_raw = float(sources_df["latitude"].min())
    max_lat_raw = float(sources_df["latitude"].max())
    min_lon_raw = float(sources_df["longitude"].min())
    max_lon_raw = float(sources_df["longitude"].max())

    lat_pad = (max_lat_raw - min_lat_raw) * padding_fraction
    lon_pad = (max_lon_raw - min_lon_raw) * padding_fraction

    # Fallback for single-point sources
    if lat_pad == 0:
        lat_pad = 0.05
    if lon_pad == 0:
        lon_pad = 0.05

    return (
        min_lat_raw - lat_pad,
        max_lat_raw + lat_pad,
        min_lon_raw - lon_pad,
        max_lon_raw + lon_pad,
    )


def latlon_to_grid(lat, lon, min_lat, max_lat, min_lon, max_lon, grid_size):
    """Convert lat/lon to grid (row, col) indices."""
    row = int((lat - min_lat) / (max_lat - min_lat) * (grid_size - 1))
    col = int((lon - min_lon) / (max_lon - min_lon) * (grid_size - 1))
    # Clamp to grid bounds
    row = max(0, min(row, grid_size - 1))
    col = max(0, min(col, grid_size - 1))
    return row, col


def add_pm10_sources(grid, sources_df, min_lat, max_lat, min_lon, max_lon, grid_size):
    """
    Add PM10 emission weights from industrial sources to the grid.
    Mirrors add_pm10_sources() from pollution_grid_v2.py.
    """
    updated = grid.copy()
    weight_col = "pm10_weight" if "pm10_weight" in sources_df.columns else None

    for _, source in sources_df.iterrows():
        row, col = latlon_to_grid(
            float(source["latitude"]), float(source["longitude"]),
            min_lat, max_lat, min_lon, max_lon, grid_size,
        )
        weight = float(source[weight_col]) if weight_col else 1.0
        if not np.isnan(weight):
            updated[row, col] += weight

    return updated


def diffuse(grid, diffusion_rate):
    """
    Apply 2D Laplacian diffusion to the pollution grid.
    Mirrors diffuse() from pollution_grid_v2.py.
    """
    updated = grid.copy()
    center      = grid[1:-1, 1:-1]
    row_plus_1  = grid[2:,   1:-1]
    row_minus_1 = grid[:-2,  1:-1]
    col_plus_1  = grid[1:-1, 2:]
    col_minus_1 = grid[1:-1, :-2]

    laplacian = (row_plus_1 + row_minus_1 + col_plus_1 + col_minus_1 - 4 * center)
    updated[1:-1, 1:-1] = center + diffusion_rate * laplacian
    return updated


def calculate_entropy(grid):
    """
    Shannon entropy of the pollution concentration grid.
    Mirrors calculate_entropy() from pollution_grid_v2.py.
    """
    total = grid.sum()
    if total == 0:
        return 0.0
    prob_grid = grid / total
    prob_vals = prob_grid[prob_grid > 0]
    return float(-np.sum(prob_vals * np.log(prob_vals)))


def run_diffusion_simulation(sources_df,
                              time_steps=DEFAULT_TIME_STEPS,
                              diffusion_rate=DEFAULT_DIFFUSION_RATE,
                              decay_rate=DEFAULT_DECAY_RATE,
                              grid_size=DEFAULT_GRID_SIZE):
    """
    Run the full PM10 diffusion simulation.

    Parameters
    ----------
    sources_df     : DataFrame from load_industry_sources() with latitude/longitude/pm10_weight
    time_steps     : number of simulation steps (default 150)
    diffusion_rate : Laplacian diffusion coefficient (default 0.15)
    decay_rate     : per-step decay multiplier (default 0.995)
    grid_size      : grid resolution (default 101 × 101)

    Returns
    -------
    final_grid    : 2D numpy array (grid_size × grid_size) — final concentration field
    entropy_list  : list of entropy values per time step
    grid_bounds   : (min_lat, max_lat, min_lon, max_lon)
    source_cells  : list of (row, col) tuples for each source
    """
    if sources_df.empty:
        raise ValueError("sources_df is empty — cannot run simulation.")

    sources_df = sources_df.dropna(subset=["latitude", "longitude"]).copy()
    min_lat, max_lat, min_lon, max_lon = build_grid_bounds(sources_df)

    grid = np.zeros((grid_size, grid_size))
    entropy_list = []

    for t in range(time_steps):
        grid = add_pm10_sources(grid, sources_df,
                                min_lat, max_lat, min_lon, max_lon, grid_size)
        grid = diffuse(grid, diffusion_rate)
        grid = grid * decay_rate
        entropy_list.append(calculate_entropy(grid))

    # Collect source cell positions for overlay
    source_cells = []
    for _, source in sources_df.iterrows():
        row, col = latlon_to_grid(
            float(source["latitude"]), float(source["longitude"]),
            min_lat, max_lat, min_lon, max_lon, grid_size,
        )
        source_cells.append((row, col))

    return grid, entropy_list, (min_lat, max_lat, min_lon, max_lon), source_cells


def get_snapshot_at_step(sources_df, target_step,
                          diffusion_rate=DEFAULT_DIFFUSION_RATE,
                          decay_rate=DEFAULT_DECAY_RATE,
                          grid_size=DEFAULT_GRID_SIZE):
    """
    Run the simulation up to target_step and return the grid snapshot.
    Useful for interactive slider-driven simulation in Streamlit.
    """
    grid, entropy_list, bounds, source_cells = run_diffusion_simulation(
        sources_df,
        time_steps=target_step,
        diffusion_rate=diffusion_rate,
        decay_rate=decay_rate,
        grid_size=grid_size,
    )
    return grid, entropy_list, bounds, source_cells


# ---------------------------------------------------------------------------
# Plume particle frame helpers
# (wraps build_particle_frames_for_record from 10_run_final_prediction_pipeline.py)
# ---------------------------------------------------------------------------

PLUME_FRAME_COUNT      = 30
PARTICLES_PER_FRAME    = 35
PLUME_STEP_SCALE       = 0.01
PARTICLE_SPREAD_SCALE  = 0.002
RANDOM_STATE           = 42

DEFAULT_SOURCE_LAT  = 26.065
DEFAULT_SOURCE_LON  = 91.875
DEFAULT_SOURCE_NAME = "Prototype Byrnihat industrial source"


def build_particle_frames_for_record(row, row_id):
    """
    Generate particle animation frames for a single prediction record.
    Preserves identical logic from ml/10_run_final_prediction_pipeline.py.

    Returns
    -------
    frames         : list of frame dicts (frame_index, center_lat, center_lon, particles)
    flat_frame_rows: list of dicts suitable for DataFrame construction
    """
    rng = np.random.default_rng(RANDOM_STATE + int(row_id))

    # Source coordinates
    source_lat = float(row.get("source_lat", DEFAULT_SOURCE_LAT))
    source_lon = float(row.get("source_lon", DEFAULT_SOURCE_LON))

    # ML-2 plume vector
    plume_u = float(row.get("ml2_predicted_plume_u", 0.0))
    plume_v = float(row.get("ml2_predicted_plume_v", 0.0))
    plume_strength = float(np.sqrt(plume_u ** 2 + plume_v ** 2))

    # Pollution-based particle styling
    pm25_val = row.get("ml1_predicted_pm25") or row.get("pm25") or 1.0
    try:
        pollution_strength = float(pm25_val) if not pd.isna(float(pm25_val)) else 1.0
    except Exception:
        pollution_strength = 1.0

    rainfall = row.get("rainfall", 0)
    try:
        rainfall = float(rainfall) if not pd.isna(float(rainfall)) else 0.0
    except Exception:
        rainfall = 0.0

    rainfall_factor = 1 / (1 + rainfall)
    particle_size   = 3.2 + min(pollution_strength / 80, 3)
    base_opacity    = 0.35 + min(pollution_strength / 300, 0.35)
    base_opacity    = max(0.12, min(base_opacity * rainfall_factor, 0.75))

    frames         = []
    flat_frame_rows = []

    for frame_index in range(PLUME_FRAME_COUNT):
        progress = frame_index / max(PLUME_FRAME_COUNT - 1, 1)

        center_lat = source_lat + plume_v * PLUME_STEP_SCALE * progress
        center_lon = source_lon + plume_u * PLUME_STEP_SCALE * progress

        spread = (PARTICLE_SPREAD_SCALE
                  * (1 + progress * 3)
                  * (1 + plume_strength * 0.05))

        particles = []
        for particle_index in range(PARTICLES_PER_FRAME):
            p_lat = center_lat + rng.normal(0, spread)
            p_lon = center_lon + rng.normal(0, spread)
            opacity = max(0.08, min(base_opacity * (1 - progress * 0.35), 0.75))

            particle = {
                "lat":     round(float(p_lat), 6),
                "lon":     round(float(p_lon), 6),
                "size":    round(float(particle_size), 3),
                "opacity": round(float(opacity), 3),
            }
            particles.append(particle)

            flat_frame_rows.append({
                "record_id":      int(row_id),
                "frame_index":    int(frame_index),
                "particle_index": int(particle_index),
                "lat":            particle["lat"],
                "lon":            particle["lon"],
                "size":           particle["size"],
                "opacity":        particle["opacity"],
                "center_lat":     round(float(center_lat), 6),
                "center_lon":     round(float(center_lon), 6),
            })

        frames.append({
            "frame_index": int(frame_index),
            "center_lat":  round(float(center_lat), 6),
            "center_lon":  round(float(center_lon), 6),
            "particles":   particles,
        })

    return frames, flat_frame_rows


def get_record_summary(final_df, record_id):
    """
    Extract a summary dict for a single plume animation record.
    Used to display the info panel alongside the animation.

    Returns a dict with source coords, ML-1/ML-2 predictions, pollution values.
    """
    if record_id >= len(final_df) or record_id < 0:
        return {}

    row = final_df.iloc[record_id]
    summary = {}

    for key in ["datetime", "station_name", "source_name", "industry_name",
                "source_lat", "source_lon",
                "ml2_predicted_plume_u", "ml2_predicted_plume_v",
                "ml2_predicted_plume_angle", "ml2_predicted_plume_strength",
                "ml1_predicted_pm25", "ml1_predicted_pm10",
                "pm25", "pm10", "wind_speed", "wind_direction",
                "temperature", "humidity", "rainfall"]:
        if key in row.index:
            val = row[key]
            if pd.isna(val):
                val = None
            elif hasattr(val, "item"):
                val = val.item()
            summary[key] = val

    return summary
