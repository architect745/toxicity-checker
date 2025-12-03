import streamlit as st
from utils import inject_css, load_artifacts

st.set_page_config(page_title="Toxicity Checker", page_icon="🧪", layout="centered")
inject_css()

model, vec, label_name = load_artifacts()

st.markdown("""
<div class="hero">
  <h2>🧪 Toxicity Checker (Mini Project)</h2>
  <p class="muted">
    Type a compound name → we fetch its SMILES from PubChem → predict toxicity (Tox21 label) → show explanation.
  </p>
  <span class="badge">Model: Logistic Regression</span>
  <span class="badge">Features: SMILES n-grams</span>
  <span class="badge">Explainable: top patterns</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("What you can do")
st.write(
    "- Go to **Predict** page and type a compound name (like ibuprofen).\n"
    "- The app shows **TOXIC / NON-TOXIC** (for one assay label).\n"
    "- It also shows **which SMILES patterns** pushed the prediction."
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Important note")
st.write(
    "This is a **dataset-based demo** (Tox21 assay label: "
    f"**{label_name}**). It is **not medical advice** and not real human safety."
)
st.markdown('</div>', unsafe_allow_html=True)

st.info("Use the left sidebar to open the other pages: Predict / Explain / About.")
