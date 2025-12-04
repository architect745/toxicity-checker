import streamlit as st
from utils import inject_css

st.set_page_config(page_title="Toxicity Checker", page_icon="🧪", layout="centered")
inject_css()

home    = st.Page("pages/1_Home.py",    title="Home",    icon="🏠")
predict = st.Page("pages/2_Predict.py", title="Predict", icon="🔮")
explain = st.Page("pages/3_Explain.py", title="Explain", icon="🧠")
about   = st.Page("pages/4_About.py",   title="About",   icon="ℹ️")

nav = st.navigation([home, predict, explain, about], position="top")
nav.run()
