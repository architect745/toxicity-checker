import streamlit as st
from utils import inject_css, load_artifacts

st.set_page_config(page_title="About", page_icon="ℹ️", layout="centered")
inject_css()

_, _, label_name = load_artifacts()

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.markdown("## About this Mini Project")
st.markdown("<div class='muted'>Explainable AI for drug toxicity prediction (demo).</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.subheader("What it does")
st.write(
    "You type a compound name. The app fetches its SMILES from PubChem, then a trained model predicts "
    f"the toxicity label **{label_name}**, and shows which SMILES patterns influenced the result."
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.subheader("How it works (simple steps)")
st.write(
    "1) Input drug/compound name\n"
    "2) PubChem API returns SMILES\n"
    "3) SMILES is converted into character n-grams\n"
    "4) Logistic Regression predicts probability\n"
    "5) Explanation shows top n-grams pushing decision up/down"
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.subheader("Limitations")
st.write(
    "- This is **dataset-based** and predicts one assay label, not full human safety.\n"
    "- PubChem lookup can fail for brand names or mixtures.\n"
    "- Explanations are text-pattern-based (simple, but interpretable)."
)
st.markdown('</div>', unsafe_allow_html=True)

# ---- Option B: professional author block ----
st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.subheader("Project Authors")
st.markdown(
    """
    <div class="muted">
      <div><b>Archit Jee</b></div>
      <div><b>Adarsh Singh</b></div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)
