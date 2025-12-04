import streamlit as st
from streamlit_lottie import st_lottie
from utils import inject_css, load_artifacts, load_lottie_url

# ---------- App config (ONLY here, not inside pages/*) ----------
st.set_page_config(page_title="Toxicity Checker", page_icon="🧪", layout="wide")
inject_css()

# Optional: hide the left sidebar completely
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Load label for display on Home
_, _, label_name = load_artifacts()

def home():
    # Optional animation (will show only if it loads)
    lottie = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_m9wro3.json")

    col1, col2 = st.columns([2.2, 1])

    with col1:
        st.markdown(
            f"""
            <div class="card fadeUp">
                <div class="hero-title">Toxicity Checker</div>

                <div class="muted">
                    Type a compound name → fetch SMILES from PubChem → predict toxicity → explain why.
                </div>

                <br>

                <span class="badge">Model: Logistic Regression</span>
                <span class="badge">Features: SMILES n-grams</span>
                <span class="badge">Explainable outp
