import streamlit as st
from utils import inject_css, load_artifacts

st.set_page_config(page_title="About", page_icon="ℹ️", layout="centered")
inject_css()

a = load_artifacts()
label_note = a.get("label_note", "ClinTox CT_TOX")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("## ℹ️ About this project")
st.markdown(
    f"""
    <div class='muted'>
    This web app predicts toxicity risk from a molecule’s structure.
    <br><br>
    <b>What it predicts:</b> {label_note}
    <br>
    <b>Input:</b> drug/chemical name → PubChem gives SMILES → model predicts probability
    <br>
    <b>Explainability:</b> shows SMILES n-grams (text patterns) that influence the result.
    <br><br>
    <b>Note:</b> This is a dataset-based prediction, not real medical advice.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)
