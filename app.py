import streamlit as st
from streamlit_lottie import st_lottie
from utils import inject_css, load_artifacts, load_lottie_url

# ---------- Background from URL ----------
def set_bg_from_url(image_url: str):
    st.markdown(
        f"""
        <style>
        /* Apply background for different Streamlit layouts/versions */
        .stApp {{
            background: url("{image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        [data-testid="stAppViewContainer"] {{
            background: url("{image_url}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* Optional: transparent header */
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0) !important;
        }}

        /* Make your cards readable on top */
        .card {{
            background: rgba(255,255,255,0.88) !important;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------- App config ----------
st.set_page_config(page_title="Toxicity Checker", page_icon="🧪", layout="wide")
inject_css()

# ✅ Put your background image URL here
BG_URL = "https://files.catbox.moe/evxdoq.png"
set_bg_from_url(BG_URL)

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
    lottie = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_m9wro3.json")

    col1, col2 = st.columns([2.2, 1])

    with col1:
        # Keep card in a single HTML block to avoid blank white rectangles
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
                <span class="badge">Explainable output</span>

                <br><br>

                <div>
                    Dataset label used: <b>{label_name}</b> (demo only; not medical advice).
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # No "Overview" text, no failure message, no empty card
        if lottie:
            st_lottie(lottie, height=220, key="home_lottie")

    st.markdown(
        """
        <div class="card fadeUp">
            <h3>Try these examples</h3>
            <div>caffeine, aspirin, ibuprofen, metformin, acetaminophen</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Top navigation ----------
pages = [
    st.Page(home, title="Home", icon="🏠", default=True),
    st.Page("pages/1_Predict.py", title="Predict", icon="🔮"),
    st.Page("pages/2_Explain.py", title="Explain", icon="🧠"),
    st.Page("pages/3_About.py", title="About", icon="ℹ️"),
]

nav = st.navigation(pages, position="top")
nav.run()
