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

        st.markdown(
            '<div class="hero-title">Toxicity Checker</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="muted">
                Type a compound name → fetch SMILES from PubChem → predict toxicity → explain why.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<span class="badge">Model: Logistic Regression</span>', unsafe_allow_html=True)
        st.markdown('<span class="badge">Features: SMILES n-grams</span>', unsafe_allow_html=True)
        st.markdown('<span class="badge">Explainable output</span>', unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.write(f"Dataset label used: **{label_name}** (demo only; not medical advice).")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card fadeUp floaty">', unsafe_allow_html=True)
        st.markdown("#### Overview")

        # Show animation only if it loads; otherwise show nothing
        if lottie:
            st_lottie(lottie, height=220, key="home_lottie")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.subheader("Try these examples")
    st.write("caffeine, aspirin, ibuprofen, metformin, acetaminophen")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------- Top navigation ----------
pages = [
    st.Page(home, title="Home", icon="🏠", default=True),
    st.Page("pages/1_Predict.py", title="Predict", icon="🔮"),
    st.Page("pages/2_Explain.py", title="Explain", icon="🧠"),
    st.Page("pages/3_About.py", title="About", icon="ℹ️"),
]

nav = st.navigation(pages, position="top")
nav.run()
