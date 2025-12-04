import streamlit as st
from utils import inject_css, load_artifacts

st.set_page_config(page_title="Home", page_icon="🏠", layout="centered")
inject_css()

_, _, label_name = load_artifacts()

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.markdown("## 🏠 Home")
st.markdown(
    f"<div class='muted'>This app predicts toxicity for <b>{label_name}</b> from SMILES strings.</div>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.markdown("### What you can do")
st.write("- **Predict**: enter a SMILES string and get toxic probability.")
st.write("- **Explain**: see global SMILES patterns the model learned (top positive/negative n-grams).")
st.write("- **About**: project details.")
st.markdown("</div>", unsafe_allow_html=True)
