import streamlit as st
from streamlit_lottie import st_lottie

from utils import inject_css, load_artifacts, load_lottie_url


st.set_page_config(page_title="Toxicity Checker", page_icon="🧪", layout="centered")
inject_css()

# Load artifacts just to confirm file exists & show label note
a = load_artifacts()
label_note = a.get("label_note", "ClinTox project")


# Lottie animation
lottie = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
cols = st.columns([1, 2], vertical_alignment="center")
with cols[0]:
    if lottie:
        st_lottie(lottie, height=140, key="lab_anim")
with cols[1]:
    st.markdown("## 🧪 Toxicity Checker")
    st.markdown(
        "<div class='muted'>Type a compound name → fetch SMILES from PubChem → predict ClinTox CT_TOX.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='pill'>{label_note}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.markdown("### How to use")
st.write(
    "Open **Predict** from the left sidebar pages. "
    "Enter a drug/chemical name (example: aspirin, ibuprofen, metformin)."
)
st.markdown("</div>", unsafe_allow_html=True)

st.info("Use the left sidebar to open the Predict page.")
