import streamlit as st
from utils import inject_css, load_artifacts, pubchem_name_to_smiles, explain_local

st.set_page_config(page_title="Predict", page_icon="🔮", layout="centered")
inject_css()

model, vec, label_name = load_artifacts()

st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.markdown("## 🔮 Predict Toxicity")
st.markdown(
    f"<div class='muted'>Label used: <b>{label_name}</b> (dataset-based, not medical advice)</div>",
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

with st.form("predict_form"):
    drug_name = st.text_input("Drug/Chemical name", value="paracetamol")
    c1, c2 = st.columns(2)
    with c1:
        threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05)
    with c2:
        topk = st.slider("Explain patterns (top k)", 5, 20, 12, 1)
    show_debug = st.checkbox("Show debug info (classes, features)")
    submit = st.form_submit_button("Predict")

if submit:
    smiles, cid, err = pubchem_name_to_smiles(drug_name)

    if err:
        st.error(err)
        if cid is not None:
            st.write("Matched CID (first):", cid)
        st.info("Try: caffeine, aspirin, ibuprofen, metformin, acetaminophen")
        st.stop()

    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.success(f"PubChem CID: {cid}")
    st.write("Canonical SMILES:")
    st.code(smiles, language="text")
    st.markdown('</div>', unsafe_allow_html=True)

    # Vectorize
    X = vec.transform([smiles])

    # --- IMPORTANT FIX: get probability for class "1" correctly ---
    if not hasattr(model, "predict_proba"):
        st.error("This model doesn't support predict_proba(). Retrain with LogisticRegression or similar.")
        st.stop()

    classes = list(getattr(model, "classes_", []))
    if len(classes) != 2:
        st.error(f"Model classes look wrong: {classes}. This usually means training went wrong (only one class).")
        st.stop()

    if 1 not in classes:
        st.error(f"Your model doesn't contain class '1' as TOXIC. classes_={classes}. Retrain with labels 0/1.")
        st.stop()

    proba = model.predict_proba(X)[0]   # [p(class0), p(class1)] but order depends on classes_
    p_toxic = float(proba[classes.index(1)])   # ✅ correct toxic probability

    pred_label = "TOXIC / POSITIVE" if p_toxic >= threshold else "NON-TOXIC / NEGATIVE"

    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        st.metric("Toxic probability", f"{p_toxic:.6f}")
    with colB:
        st.metric("Prediction", pred_label)
    st.progress(int(round(p_toxic * 100)))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.subheader("Explanation (local)")
    st.write("Positive contribution pushes toward TOXIC. Negative pushes toward NON-TOXIC.")
    rows = explain_local(smiles, model, vec, topk=topk)
    st.dataframe(rows, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if show_debug:
        with st.expander("Debug details"):
            st.write("model.classes_:", classes)
            st.write("Matched n-gram features (X.nnz):", int(X.nnz))
