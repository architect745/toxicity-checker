import streamlit as st
from utils import inject_css, load_artifacts

st.set_page_config(page_title="About", page_icon="ℹ️", layout="centered")
inject_css()

_, _, label_name = load_artifacts()

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.markdown("## ℹ️ About")
st.markdown(
    f"<div class='muted'>Project: Toxicity Checker for <b>{label_name}</b> (SMILES-based ML model).</div>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.markdown("### Notes")
st.write("- Predictions depend on your training data + model choice.")
st.write("- The **Explain** page shows *global* patterns (model weights), not molecule-specific attribution.")
st.write("- If your model is not linear, global n-gram weights won’t exist.")
st.markdown("</div>", unsafe_allow_html=True)
