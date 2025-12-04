import streamlit as st

from utils import (
    inject_css,
    load_artifacts,
    name_to_smiles,
    explain_local,
    UTILS_VERSION,
)

st.set_page_config(page_title="Predict", page_icon="🧪", layout="centered")
inject_css()

# Load trained artifacts
a = load_artifacts()
vec = a["vectorizer"]
tox_model = a["tox_model"]
fda_model = a.get("fda_model", None)
default_t = float(a.get("best_threshold", 0.50))
label_note = a.get("label_note", "ClinTox CT_TOX")


# ---------- Header ----------
st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
st.markdown("## 🧪 Predict Toxicity (ClinTox)")
st.markdown(
    f"<div class='muted'><b>Label:</b> {label_note}<br>"
    f"<b>Utils version:</b> {UTILS_VERSION}</div>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

st.info(
    "This is a dataset-based toxicity risk guess (clinical-trial toxicity label), not real medical advice."
)

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🔎 Drug name → SMILES", "🧬 Paste SMILES"])

smiles = None
smiles_source = None

with tab1:
    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    name = st.text_input(
        "Enter drug / chemical name",
        value="ibuprofen",
        help="Use generic names like: aspirin, ibuprofen, acetaminophen/paracetamol, caffeine, metformin."
    )
    fetch = st.button("Fetch SMILES")
    st.markdown("</div>", unsafe_allow_html=True)

    if fetch:
        s, src, err = name_to_smiles(name)
        if err:
            st.error(err)
        else:
            smiles = s
            smiles_source = src
            st.success(f"SMILES fetched via: {smiles_source}")
            st.code(smiles, language="text")

with tab2:
    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    s2 = st.text_area(
        "Paste SMILES here",
        value="",
        height=120,
        placeholder="Example: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O  (ibuprofen)"
    )
    use_smiles = st.button("Use this SMILES")
    st.markdown("</div>", unsafe_allow_html=True)

    if use_smiles:
        if not s2.strip():
            st.error("Paste a SMILES string first.")
        else:
            smiles = max(s2.strip().split("."), key=len)  # keep biggest fragment
            smiles_source = "Manual SMILES"
            st.success("Using your SMILES")
            st.code(smiles, language="text")


# ---------- Controls ----------
st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
colA, colB = st.columns(2)
with colA:
    threshold = st.slider("Decision threshold (CT_TOX)", 0.05, 0.95, default_t, 0.05)
with colB:
    topk = st.slider("Explain patterns (top k)", 5, 25, 12, 1)

show_debug = st.checkbox("Show debug info", value=False)
predict_btn = st.button("Predict Toxicity")
st.markdown("</div>", unsafe_allow_html=True)


# ---------- Predict ----------
if predict_btn:
    if not smiles:
        st.error("First fetch SMILES by name OR paste a SMILES.")
        st.stop()

    # Vectorize
    X = vec.transform([smiles])

    # Safety checks
    if X.nnz == 0:
        st.error(
            "Vectorizer found zero matching n-grams in this SMILES. "
            "This usually happens when the SMILES is very unusual or the model files don’t match."
        )
        st.stop()

    # Probabilities (IMPORTANT: choose class-1 correctly)
    classes = list(getattr(tox_model, "classes_", [0, 1]))
    if 1 not in classes or 0 not in classes:
        st.error(f"Model classes look wrong: {classes}. Retrain with labels 0/1.")
        st.stop()

    tox_proba_vec = tox_model.predict_proba(X)[0]
    p_tox = float(tox_proba_vec[classes.index(1)])  # ✅ class 1 probability

    toxic_flag = (p_tox >= threshold)

    # Optional FDA model
    p_fda = None
    if fda_model is not None and hasattr(fda_model, "predict_proba"):
        try:
            fda_classes = list(getattr(fda_model, "classes_", [0, 1]))
            fda_proba_vec = fda_model.predict_proba(X)[0]
            if 1 in fda_classes:
                p_fda = float(fda_proba_vec[fda_classes.index(1)])
        except Exception:
            p_fda = None

    # ---------- Results card ----------
    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.subheader("Result")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("CT_TOX probability", f"{p_tox:.6f}")
    with c2:
        st.metric("Prediction", "TOXIC / POSITIVE" if toxic_flag else "NON-TOXIC / NEGATIVE")

    st.progress(int(round(p_tox * 100)))

    if p_fda is not None:
        st.caption(f"Extra info: FDA_APPROVED probability (separate model): {p_fda:.4f}")

    st.caption(f"SMILES source: {smiles_source or 'Unknown'}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Explanation ----------
    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.subheader("Explanation (local)")
    st.write("These are SMILES n-grams that pushed the prediction up (positive) or down (negative).")
    exp_df = explain_local(smiles, tox_model, vec, topk=topk)
    st.dataframe(exp_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Debug ----------
    if show_debug:
        st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
        st.subheader("Debug")
        st.write("tox_model.classes_:", classes)
        st.write("Vectorizer non-zero features (X.nnz):", int(X.nnz))
        st.write("Threshold:", threshold)
        st.write("SMILES used:")
        st.code(smiles)
        st.markdown("</div>", unsafe_allow_html=True)
