import streamlit as st
from utils import inject_css, load_artifacts, top_global_ngrams

st.set_page_config(page_title="Explain", page_icon="🧠", layout="centered")
inject_css()

model, vec, label_name = load_artifacts()

st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown("## 🧠 Model Explanation (Global)")
st.markdown(f"<p class='muted'>These are the strongest SMILES patterns the model learned for <b>{label_name}</b>.</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

pos, neg = top_global_ngrams(model, vec, k=15)

c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Patterns that increase TOXIC")
    st.dataframe(pos, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Patterns that decrease TOXIC")
    st.dataframe(neg, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
