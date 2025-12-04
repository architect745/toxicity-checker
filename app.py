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

# Home page UI (animated)
def home():
    lottie = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_m9wro3.json")

    col1, col2 = st.columns([2.2, 1])

    with col1:
        st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
        st.markdown('<div class="hero-tit
