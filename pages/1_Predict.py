import streamlit as st
from utils import inject_css, load_artifacts, pubchem_name_to_smiles, explain_local

st.set_page_config(page_title="Predict", page_icon="🔮", layout="centered")
inject_css()

model, vec, label_name = load_artifacts()

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.markdown("## 🔮 Predict Toxicity")
st.markdown(f"<div class='muted'>Task label: <b>{label_name}</b> (demo label)</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

with st.form("predict_form"):
    drug_name = st.text_input("Drug/Chemical name", value="ibuprofen")
    c1, c2 = st.columns(2)
    with c1:
        threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05)
    with c2:
        topk = st.slider("Explain patterns (top k)", 5, 20, 12, 1)
    submit = st.form_submit_button("Predict")

if submit:
    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.write("Looking up PubChem…")
    st.markdown('</div>', unsafe_allow_html=True)

    smiles, cid, err = pubchem_name_to_smiles(drug_name)
    if err:
        st.error(err)
        if cid is not None:
            st.write("Matched CID (first):", cid)
        st.info("Try: caffeine, aspirin, ibuprofen, metformin, acetaminophen")
        st.stop()

    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.success(f"PubChem CID: {cid}")
    st.code(smiles, language="text")
    st.markdown('</div>', unsafe_allow_html=True)

    X = vec.transform([smiles])
    p = float(model.predict_proba(X)[:, 1][0])

    pred_label = "TOXIC / POSITIVE" if p >= threshold else "NON-TOXIC / NEGATIVE"

    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        st.metric("Predicted probability", f"{p:.3f}")
    with colB:
        st.metric("Prediction", pred_label)
    st.progress(int(round(p * 100)))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.subheader("Explanation (local)")
    st.write("Positive contribution pushes toward toxic. Negative pushes toward non-toxic.")
    rows = explain_local(smiles, model, vec, topk=topk)
    st.dataframe(rows, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
