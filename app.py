import streamlit as st
from utils import inject_css

st.set_page_config(page_title="Toxicity Checker", page_icon="🧪", layout="centered")
inject_css()

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.markdown("## 🧪 Toxicity Checker (SR-p53)")
st.markdown(
    "<div class='muted'>Use the sidebar to open pages: Home, Predict, Explain, About.</div>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

st.info("If you don't see the sidebar, click the arrow (>) on the top-left.")
