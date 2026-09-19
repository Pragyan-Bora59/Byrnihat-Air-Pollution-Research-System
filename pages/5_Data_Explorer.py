"""
pages/5_Data_Explorer.py — Dataset Explorer

Displays:
  - Dataset selector (ML-1, ML-2, Final predictions, Plume frames, AQICN)
  - Column filter
  - Searchable / filterable dataframe
  - Column statistics
  - Download button
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data_loader import (
    load_ml1_dataset,
    load_ml2_dataset,
    load_final_predictions,
    load_plume_frames,
    load_aqicn_combined,
    load_raw_city_data,
    load_industry_sources,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Data Explorer — Byrnihat",
    page_icon="🗂️",
    layout="wide",
)

CSS = Path(__file__).parents[1] / "assets" / "style.css"
if CSS.exists():
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

st.markdown("## 🗂️ Data Explorer")
st.markdown(
    "<div style='color:#94a3b8; margin-bottom:1.5rem'>"
    "Browse, filter, search and download any dataset used by the pipeline."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "Final Predictions (ML-1 + ML-2)": {
        "loader": load_final_predictions,
        "description": "Full pipeline output: recorded + ML-1 predicted pollutants + ML-2 plume direction. "
                       "Produced by ml/10_run_final_prediction_pipeline.py",
        "file": "outputs/reports/final_prediction_output.csv",
    },
    "ML-1 Training Dataset": {
        "loader": load_ml1_dataset,
        "description": "Hourly environment + pollution dataset used to train ML-1 models. "
                       "Produced by ml/06_prepare_ml1_dataset.py",
        "file": "data/processed/ml1_environment_pollution_dataset.csv",
    },
    "ML-2 Plume Training Dataset": {
        "loader": load_ml2_dataset,
        "description": "Source-to-grid plume training dataset used for ML-2. "
                       "Produced by ml/08_build_ml2_training_dataset.py",
        "file": "data/processed/ml2_plume_training_dataset.csv",
    },
    "Plume Animation Frames": {
        "loader": load_plume_frames,
        "description": "Flat particle frame data for all plume animation records. "
                       "Produced by ml/10_run_final_prediction_pipeline.py",
        "file": "outputs/maps/ml_predicted_plume_frames.csv",
    },
    "AQICN Combined Observed": {
        "loader": load_aqicn_combined,
        "description": "Cleaned AQICN observed data for Byrnihat, Guwahati and Shillong. "
                       "Produced by analysis/observed_data_analysis_v1.py",
        "file": "outputs/observed_aqicn_combined_cleaned.csv",
    },
    "Raw City AQICN Data": {
        "loader": load_raw_city_data,
        "description": "Raw AQICN CSV files for all three cities combined.",
        "file": "data/{byrnihat,guwahati,shillong}_aqicn_data.csv",
    },
    "Industry Sources": {
        "loader": load_industry_sources,
        "description": "Active industrial emission sources with coordinates, "
                       "pollutant weights and terrain/elevation data.",
        "file": "outputs/industry_elevation_table.csv or data/byrnihat_33_active_sources_*.csv",
    },
}


# ---------------------------------------------------------------------------
# Dataset selector
# ---------------------------------------------------------------------------

selected_ds = st.selectbox(
    "Select dataset",
    options=list(DATASET_REGISTRY.keys()),
    key="ds_selector",
)

info = DATASET_REGISTRY[selected_ds]
with st.expander(f"ℹ️ About: {selected_ds}", expanded=False):
    st.markdown(f"**Description:** {info['description']}")
    st.markdown(f"**Source file(s):** `{info['file']}`")

# Load selected dataset
with st.spinner(f"Loading {selected_ds}…"):
    try:
        df = info["loader"]()
        load_ok = True
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        load_ok = False
        df = pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        load_ok = False
        df = pd.DataFrame()

if not load_ok or df.empty:
    st.warning("No data available for this dataset. Check that the pipeline has been run.")
    st.stop()

# Datetime parsing for any datetime column
for dt_col in ["datetime", "api_time"]:
    if dt_col in df.columns:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        break

st.caption(
    f"✅ Loaded **{len(df):,} rows × {len(df.columns)} columns** "
    f"from **{selected_ds}**"
)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

with st.expander("🔧 Column & Row Filters", expanded=True):
    fc1, fc2 = st.columns([2, 1])

    with fc1:
        all_cols = df.columns.tolist()
        selected_cols = st.multiselect(
            "Show columns",
            options=all_cols,
            default=all_cols[:min(15, len(all_cols))],
            key="ds_col_filter",
        )

    with fc2:
        n_rows = st.number_input(
            "Max rows to display",
            min_value=10,
            max_value=50000,
            value=500,
            step=100,
            key="ds_max_rows",
        )

    # Optional column-based value filter
    fc3, fc4, fc5 = st.columns(3)
    with fc3:
        filter_col = st.selectbox(
            "Filter by column",
            options=["(none)"] + all_cols,
            key="ds_filter_col",
        )
    with fc4:
        filter_op = st.selectbox(
            "Operator",
            options=["contains", "==", ">=", "<=", ">", "<"],
            key="ds_filter_op",
        )
    with fc5:
        filter_val = st.text_input("Value", key="ds_filter_val")

# Apply column selection
display_cols = selected_cols if selected_cols else all_cols
view_df = df[display_cols].copy()

# Apply row filter
if filter_col != "(none)" and filter_col in view_df.columns and filter_val.strip():
    col_series = view_df[filter_col]
    try:
        if filter_op == "contains":
            mask = col_series.astype(str).str.contains(filter_val, case=False, na=False)
        else:
            num_val = float(filter_val)
            ops = {"==": col_series == num_val, ">=": col_series >= num_val,
                   "<=": col_series <= num_val, ">":  col_series > num_val,
                   "<":  col_series < num_val}
            mask = ops[filter_op]
        view_df = view_df[mask]
        st.caption(f"Filter applied: **{filter_col} {filter_op} '{filter_val}'** → "
                   f"{len(view_df):,} rows remaining")
    except Exception as e:
        st.warning(f"Could not apply filter: {e}")


# ---------------------------------------------------------------------------
# Main dataframe display
# ---------------------------------------------------------------------------

st.dataframe(
    view_df.head(n_rows).round(6),
    use_container_width=True,
    hide_index=True,
)

st.caption(f"Displaying first {min(n_rows, len(view_df)):,} of {len(view_df):,} filtered rows")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

dl_cols = st.columns(2)
with dl_cols[0]:
    st.download_button(
        "⬇️ Download displayed data (CSV)",
        data=view_df.head(n_rows).to_csv(index=False),
        file_name=f"{selected_ds.replace(' ', '_').lower()}_export.csv",
        mime="text/csv",
        key="ds_dl_displayed",
    )
with dl_cols[1]:
    st.download_button(
        "⬇️ Download full filtered data (CSV)",
        data=view_df.to_csv(index=False),
        file_name=f"{selected_ds.replace(' ', '_').lower()}_full.csv",
        mime="text/csv",
        key="ds_dl_full",
    )


# ---------------------------------------------------------------------------
# Column statistics
# ---------------------------------------------------------------------------

with st.expander("📊 Column Statistics", expanded=False):
    stat_cols = view_df.select_dtypes(include="number").columns.tolist()

    if stat_cols:
        desc = view_df[stat_cols].describe().T.round(4)
        desc.index.name = "column"
        desc = desc.reset_index()
        st.dataframe(desc, use_container_width=True, hide_index=True)
    else:
        st.info("No numeric columns in the current column selection.")

    # Non-numeric overview
    obj_cols = view_df.select_dtypes(exclude="number").columns.tolist()
    if obj_cols:
        st.markdown("**Non-numeric column overview:**")
        for c in obj_cols[:10]:  # limit to 10
            n_unique = view_df[c].nunique()
            n_null   = view_df[c].isna().sum()
            st.markdown(
                f"- **{c}**: {n_unique} unique values, {n_null} nulls"
            )


# ---------------------------------------------------------------------------
# Data shape and dtype summary
# ---------------------------------------------------------------------------

with st.expander("📐 Schema / dtypes", expanded=False):
    schema = pd.DataFrame({
        "column": df.columns,
        "dtype":  df.dtypes.astype(str).values,
        "non_null": df.notna().sum().values,
        "null_pct": (df.isna().mean() * 100).round(1).values,
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)
