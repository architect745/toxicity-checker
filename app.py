import streamlit as st
from streamlit_lottie import st_lottie
from utils import inject_css, load_artifacts, load_lottie_url

st.set_page_config(page_title="Toxicity Checker", page_icon="🧪", layout="centered")
inject_css()

# load artifacts just to show the label on home page
model, vec, label_name = load_artifacts()

# Lottie animation (nice lab vibe)
lottie = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_m9wro3.json")

col1, col2 = st.columns([2.2, 1])

with col1:
    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">🧪 Toxicity Checker</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='muted'>Type a compound name → fetch SMILES from PubChem → predict toxicity → explain why.</div>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="badge">Model: Logistic Regression</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge">Features: SMILES n-grams</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge">Explainable output</span>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.write(f"Dataset label used: **{label_name}** (demo only, not medical advice).")
    st.info("Use the left sidebar to open: Predict / Explain / About.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card fadeUp floaty">', unsafe_allow_html=True)
    st.markdown("#### Animation")
    if lottie:
        st_lottie(lottie, height=220, key="home_lottie")
    else:
        st.write("Animation failed to load (internet blocked).")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.subheader("Quick demo inputs (try these)")
st.write("caffeine, aspirin, ibuprofen, metformin, acetaminophen")
st.markdown('</div>', unsafe_allow_html=True)
