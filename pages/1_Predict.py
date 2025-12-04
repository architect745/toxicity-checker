import streamlit as st
from utils import inject_css, load_artifacts, predict_smiles

st.set_page_config(page_title="Predict", page_icon="🔮", layout="centered")
inject_css()

model, vec, label_name = load_artifacts()

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.markdown("## 🔮 Predict")
st.markdown(
    f"<div class='muted'>Enter a SMILES string to predict <b>{label_name}</b>.</div>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

smiles = st.text_area("SMILES", placeholder="Example: CC(=O)Oc1ccccc1C(=O)O", height=120)

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("Predict", use_container_width=True)
with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.rerun()

if run:
    if not smiles.strip():
        st.error("Give a SMILES string. Empty input = no prediction.")
    else:
        try:
            p = predict_smiles(smiles.strip(), model, vec)
            st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
            st.metric("Toxic probability", f"{p:.4f}")
            st.write("Decision rule (simple):")
            st.write("- **Toxic** if probability ≥ 0.50")
            st.write("- **Non-toxic** otherwise")
            st.success("Toxic" if p >= 0.50 else "Non-toxic")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
