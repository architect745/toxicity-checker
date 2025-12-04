import streamlit as st
from utils import inject_css, load_artifacts, top_global_ngrams

st.set_page_config(page_title="Explain", page_icon="📌", layout="centered")
inject_css()

a = load_artifacts()
vec = a["vectorizer"]
tox_model = a["tox_model"]
label_note = a.get("label_note", "ClinTox CT_TOX")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("## 📌 Explain (Global)")
st.markdown(
    "<div class='muted'>These are the SMILES n-gram patterns that most influence the model overall.</div>",
    unsafe_allow_html=True
)
st.markdown(f"<div class='muted'><b>Model:</b> {label_note}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

topk = st.slider("How many patterns to show", 5, 40, 20, 5)

pos_df, neg_df = top_global_ngrams(tox_model, vec, topk=topk)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Patterns that increase TOXIC")
st.dataframe(pos_df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Patterns that decrease TOXIC")
st.dataframe(neg_df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
