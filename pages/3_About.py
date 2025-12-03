import streamlit as st
from utils import inject_css, load_artifacts

st.set_page_config(page_title="About", page_icon="ℹ️", layout="centered")
inject_css()

_, _, label_name = load_artifacts()

st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown("## ℹ️ About this Mini Project")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Goal")
st.write("Build an **Explainable AI** demo for drug discovery: predict a toxicity assay label and explain the decision.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("How it works (simple)")
st.write(
    "1) User types compound name\n"
    "2) App fetches SMILES from PubChem\n"
    "3) Convert SMILES → character n-grams\n"
    "4) Logistic Regression predicts probability\n"
    "5) Show top n-grams that influenced the output"
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Limitations (say this to teacher)")
st.write(
    f"- Prediction is for **one dataset label** ({label_name}), not real human safety.\n"
    "- Explanations are SMILES text patterns; they are **simple but interpretable**.\n"
    "- PubChem name matching can fail for brand names or mixtures."
)
st.markdown('</div>', unsafe_allow_html=True)
