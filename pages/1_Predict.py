import streamlit as st
from utils import inject_css, load_artifacts, pubchem_name_to_smiles, explain_local

st.set_page_config(page_title="Predict", page_icon="🧪", layout="centered")
inject_css()

a = load_artifacts()
vec = a["vectorizer"]
tox_model = a["tox_model"]
fda_model = a["fda_model"]
default_t = float(a.get("best_threshold", 0.5))
label_note = a.get("label_note", "ClinTox models")

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.markdown("## 🧪 Toxicity Checker (ClinTox)")
st.markdown(f"<div class='muted'>{label_note}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Search by Drug Name", "Paste SMILES"])

smiles = None
cid = None

with tab1:
    name = st.text_input("Drug/Chemical name", value="paracetamol")
    if st.button("Fetch SMILES from PubChem"):
        s, c, err = pubchem_name_to_smiles(name)
        if err:
            st.error(err)
        else:
            smiles, cid = s, c
            st.success(f"Found PubChem CID: {cid}")
            st.code(smiles)

with tab2:
    s2 = st.text_area("SMILES", value="", height=90, placeholder="Paste a SMILES string here…")
    if st.button("Use this SMILES"):
        if not s2.strip():
            st.error("Paste a SMILES first.")
        else:
            smiles = max(s2.strip().split("."), key=len)
            st.success("Using your SMILES:")
            st.code(smiles)

st.markdown("---")
threshold = st.slider("Decision threshold (for CT_TOX)", 0.05, 0.95, default_t, 0.05)
topk = st.slider("Explain patterns (top k)", 5, 20, 12, 1)

if st.button("Predict"):
    if not smiles:
        st.error("First get a SMILES (by name or paste SMILES).")
        st.stop()

    X = vec.transform([smiles])

    # probabilities
    p_tox = float(tox_model.predict_proba(X)[0][1])  # class 1 prob
    p_fda = float(fda_model.predict_proba(X)[0][1])  # class 1 prob

    toxic_flag = (p_tox >= threshold)

    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.metric("CT_TOX probability (trial-tox risk)", f"{p_tox:.4f}")
    st.metric("FDA_APPROVED probability", f"{p_fda:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    if toxic_flag:
        st.error("Prediction: TOXIC (CT_TOX positive)")
    else:
        st.success("Prediction: NON-TOXIC (CT_TOX negative)")
    st.markdown("<div class='muted'>This is a dataset-based risk guess, not medical advice.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.subheader("Why it predicted that (pattern explanation)")
    st.write("Positive contributions push toward TOXIC; negative push toward NON-TOXIC.")
    st.dataframe(explain_local(smiles, tox_model, vec, topk=topk), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
