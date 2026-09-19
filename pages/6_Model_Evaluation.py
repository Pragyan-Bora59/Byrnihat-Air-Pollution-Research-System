"""
pages/6_Model_Evaluation.py — Model Evaluation

Displays:
  - Feature Importance (ML-1, ML-2)
  - Detailed Metrics (RMSE, MAE, R²)
  - Test Set vs Final Prediction metrics comparison
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data_loader import (
    load_ml1_metrics,
    load_ml2_metrics,
    load_final_predictions,
)
from backend.chart_builder import plot_feature_importance


st.set_page_config(
    page_title="Model Evaluation — Byrnihat",
    page_icon="📐",
    layout="wide",
)

CSS = Path(__file__).parents[1] / "assets" / "style.css"
if CSS.exists():
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

st.markdown("## 📐 Model Evaluation")
st.markdown(
    "<div style='color:#94a3b8; margin-bottom:1.5rem'>"
    "Detailed evaluation metrics and feature importance for ML-1 and ML-2 models."
    "</div>",
    unsafe_allow_html=True,
)

# Load metrics
try:
    ml1_metrics = load_ml1_metrics()
    ml2_metrics = load_ml2_metrics()
    pred_df     = load_final_predictions()
    ok = True
except FileNotFoundError as e:
    st.error(f"Missing data: {e}")
    ok = False

if not ok:
    st.stop()


# ---------------------------------------------------------------------------
# ML-1 Evaluation
# ---------------------------------------------------------------------------

st.markdown("### ML-1: Environment & Pollution Model")

with st.expander("📊 ML-1 Test Set Metrics", expanded=True):
    if not ml1_metrics.empty:
        # Separate overall metrics from feature importance
        m_df = ml1_metrics[~ml1_metrics["target"].str.contains("feature_importance", na=False)].copy()
        
        # Display as a dataframe
        st.dataframe(
            m_df.style.format({"rmse": "{:.4f}", "mae": "{:.4f}", "r2": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No ML-1 metrics found.")

# ML-1 Feature importance
st.markdown("#### Feature Importance (Top Features)")
if not ml1_metrics.empty:
    fi_df = ml1_metrics[ml1_metrics["target"].str.contains("feature_importance", na=False)]
    if not fi_df.empty:
        # Reformat it into a usable shape
        fi_clean = []
        for _, row in fi_df.iterrows():
            target = str(row["target"]).replace("_feature_importance", "")
            feat   = str(row["metric"])
            try:
                val = float(row["value"])
            except:
                val = 0.0
            fi_clean.append({"target": target, "feature": feat, "importance": val})
            
        fi_clean_df = pd.DataFrame(fi_clean)
        
        targets = fi_clean_df["target"].unique()
        sel_target = st.selectbox("Select target to view feature importance", options=targets, key="ml1_fi")
        
        subset = fi_clean_df[fi_clean_df["target"] == sel_target].sort_values("importance", ascending=False)
        fig_fi = plot_feature_importance(subset, "importance", "feature", f"Feature Importance for {sel_target}")
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("No feature importance data found for ML-1.")


# ---------------------------------------------------------------------------
# ML-2 Evaluation
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### ML-2: Plume Vector Model")

with st.expander("📊 ML-2 Test Set Metrics", expanded=True):
    if not ml2_metrics.empty:
        m2_df = ml2_metrics[~ml2_metrics["target"].str.contains("feature_importance", na=False)].copy()
        st.dataframe(
            m2_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No ML-2 metrics found.")

# ML-2 Feature importance
st.markdown("#### Feature Importance")
if not ml2_metrics.empty:
    fi2_df = ml2_metrics[ml2_metrics["target"].str.contains("feature_importance", na=False)]
    if not fi2_df.empty:
        fi2_clean = []
        for _, row in fi2_df.iterrows():
            target = str(row["target"]).replace("_feature_importance", "")
            feat   = str(row["metric"])
            try:
                val = float(row["value"])
            except:
                val = 0.0
            fi2_clean.append({"target": target, "feature": feat, "importance": val})
            
        fi2_clean_df = pd.DataFrame(fi2_clean)
        
        targets2 = fi2_clean_df["target"].unique()
        sel_target2 = st.selectbox("Select target", options=targets2, key="ml2_fi")
        
        subset2 = fi2_clean_df[fi2_clean_df["target"] == sel_target2].sort_values("importance", ascending=False)
        fig_fi2 = plot_feature_importance(subset2, "importance", "feature", f"Feature Importance for {sel_target2}")
        st.plotly_chart(fig_fi2, use_container_width=True)
    else:
        st.info("No feature importance data found for ML-2.")
