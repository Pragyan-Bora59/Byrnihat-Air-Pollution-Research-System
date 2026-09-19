"""
app.py — Byrnihat Air Pollution Research System
Streamlit entry point.

Responsibilities (ONLY):
  - Page configuration
  - Custom CSS injection
  - Sidebar navigation and project info
  - Data status indicator

NO simulation, ML, or data-processing logic belongs here.
"""

from pathlib import Path
import streamlit as st


# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Byrnihat Air Pollution Research System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "**Byrnihat Air Pollution Research System**\n\n"
            "Research-grade ML-driven air quality prediction and simulation "
            "for the Byrnihat industrial corridor, Assam–Meghalaya."
        ),
    },
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CSS_FILE = Path(__file__).parent / "assets" / "style.css"
if CSS_FILE.exists():
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Project title
    st.markdown(
        """
        <div style='text-align:center; padding:0.5rem 0 1rem'>
            <div style='font-size:2.2rem'>🏭</div>
            <div style='font-size:1.05rem; font-weight:700;
                        color:#60a5fa; letter-spacing:0.02em;
                        line-height:1.3'>
                Byrnihat Air<br>Pollution System
            </div>
            <div style='font-size:0.72rem; color:#94a3b8;
                        margin-top:0.3rem; letter-spacing:0.04em'>
                Assam – Meghalaya Border
            </div>
        </div>
        <hr style='border:none; border-top:1px solid #334155; margin:0 0 1rem'>
        """,
        unsafe_allow_html=True,
    )

    # Navigation label
    st.markdown(
        "<div style='font-size:0.72rem; color:#94a3b8; "
        "text-transform:uppercase; letter-spacing:0.08em;"
        "margin-bottom:0.4rem'>Navigation</div>",
        unsafe_allow_html=True,
    )

    # Data status indicator
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem; color:#94a3b8; "
        "text-transform:uppercase; letter-spacing:0.08em;"
        "margin-bottom:0.4rem'>Data Status</div>",
        unsafe_allow_html=True,
    )

    try:
        from backend.data_loader import check_file_status
        statuses = check_file_status()
        all_ok = all(v["exists"] for v in statuses.values())
        n_ok   = sum(v["exists"] for v in statuses.values())
        n_total = len(statuses)

        if all_ok:
            st.success(f"✅ All {n_total} data files found")
        else:
            st.warning(f"⚠️ {n_ok}/{n_total} data files found")

        with st.expander("File details", expanded=False):
            for name, info in statuses.items():
                icon = "✅" if info["exists"] else "❌"
                size = f"{info['size_mb']} MB" if info["exists"] else "missing"
                st.markdown(
                    f"<div style='font-size:0.78rem; margin-bottom:2px;'>"
                    f"{icon} <b>{name}</b><br>"
                    f"<span style='color:#94a3b8'>{size}</span></div>",
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"Could not check data files:\n{e}")

    # Model info
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem; color:#94a3b8; "
        "text-transform:uppercase; letter-spacing:0.08em;"
        "margin-bottom:0.4rem'>ML Models</div>",
        unsafe_allow_html=True,
    )

    project_root = Path(__file__).parent
    models_dir   = project_root / "models"

    if models_dir.exists():
        model_files = list(models_dir.glob("*.joblib"))
        ml1_count   = sum(1 for m in model_files if "ml1" in m.name)
        ml2_count   = sum(1 for m in model_files if "ml2" in m.name)
        st.markdown(
            f"<div style='font-size:0.82rem; color:#f1f5f9'>"
            f"🤖 ML-1: <b>{ml1_count}</b> models<br>"
            f"🧭 ML-2: <b>{ml2_count}</b> model(s)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='font-size:0.82rem; color:#94a3b8'>models/ not found</div>",
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        """
        <div style='position:absolute; bottom:1rem; left:1rem; right:1rem;
                    text-align:center; font-size:0.7rem; color:#475569'>
            Research prototype<br>
            Byrnihat, Assam–Meghalaya
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style='padding: 2rem 0 1rem'>
        <h1 style='font-size:2.4rem; font-weight:800; color:#f1f5f9;
                   line-height:1.2; margin-bottom:0.5rem'>
            🏭 Byrnihat Air Pollution<br>Research System
        </h1>
        <p style='font-size:1.05rem; color:#94a3b8; max-width:700px;
                  line-height:1.7; margin-bottom:1.5rem'>
            Research-grade ML-driven air quality prediction and plume dispersion
            simulation for the Byrnihat industrial corridor on the
            Assam–Meghalaya border.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick navigation cards
col1, col2, col3, col4 = st.columns(4)

nav_cards = [
    ("📊", "Dashboard",          "Latest metrics, AQI status and model summary",    "pages/1_Dashboard.py"),
    ("🌫️", "Simulation",         "Plume animation, diffusion grid and wind vectors", "pages/2_Simulation.py"),
    ("🤖", "ML Predictions",     "ML-1 pollution / weather · ML-2 plume direction",  "pages/3_ML_Predictions.py"),
    ("📈", "Historical Analysis", "Multi-city pollution trends over time",            "pages/4_Historical_Analysis.py"),
]

for col, (icon, title, desc, _) in zip([col1, col2, col3, col4], nav_cards):
    with col:
        st.markdown(
            f"""
            <div style='background:#1e293b; border:1px solid #334155;
                        border-radius:12px; padding:1.25rem; height:160px;
                        transition:transform 0.2s;'>
                <div style='font-size:1.8rem; margin-bottom:0.5rem'>{icon}</div>
                <div style='font-size:1rem; font-weight:700; color:#f1f5f9;
                            margin-bottom:0.4rem'>{title}</div>
                <div style='font-size:0.82rem; color:#94a3b8;
                            line-height:1.5'>{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

col5, col6, col7 = st.columns(3)
nav_cards2 = [
    ("🗂️", "Data Explorer",    "Browse, filter and download datasets"),
    ("📐", "Model Evaluation", "RMSE · MAE · R² · Feature importance"),
    ("ℹ️", "About",            "Methodology · Data sources · Architecture"),
]
for col, (icon, title, desc) in zip([col5, col6, col7], nav_cards2):
    with col:
        st.markdown(
            f"""
            <div style='background:#1e293b; border:1px solid #334155;
                        border-radius:12px; padding:1.25rem; height:140px;'>
                <div style='font-size:1.8rem; margin-bottom:0.5rem'>{icon}</div>
                <div style='font-size:1rem; font-weight:700; color:#f1f5f9;
                            margin-bottom:0.4rem'>{title}</div>
                <div style='font-size:0.82rem; color:#94a3b8;
                            line-height:1.5'>{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# System overview
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-header'>System Architecture</div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        **ML-1 — Pollution & Weather Prediction**
        - ExtraTreesRegressor pipeline
        - 11 targets: PM2.5, PM10, SO2, CO, NO2, O3, NH3, NO, NOx, Wind U, Wind V
        - NASA POWER + AQICN features
        """,
    )
with c2:
    st.markdown(
        """
        **ML-2 — Plume Direction Prediction**
        - Multi-output ExtraTreesRegressor
        - Predicts plume U/V vectors → angle + strength
        - Source-to-grid spatial features
        """,
    )
with c3:
    st.markdown(
        """
        **Simulation Engine**
        - 2D Laplacian PM10 diffusion grid
        - 150 time steps, Gaussian particle plume
        - DEM terrain hillshade overlay
        """,
    )

st.markdown("---")
st.markdown(
    "<div style='font-size:0.78rem; color:#475569; text-align:center'>"
    "Use the sidebar pages to navigate between sections. "
    "All outputs are pre-computed — no model inference runs in the UI."
    "</div>",
    unsafe_allow_html=True,
)
