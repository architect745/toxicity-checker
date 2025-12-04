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

# ---- keep values between clicks ----
if "smiles" not in st.session_state:
    st.session_state.smiles = None
if "smiles_source" not in st.session_state:
    st.session_state.smiles_source = None

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

st.info("This is dataset-based risk prediction (clinical trial toxicity label), not medical advice.")

# Show currently saved SMILES (if any)
if st.session_state.smiles:
    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.success(f"✅ Using SMILES from: {st.session_state.smiles_source}")
    st.code(st.session_state.smiles, language="text")
    if st.button("Clear saved SMILES"):
        st.session_state.smiles = None
        st.session_state.smiles_source = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🔎 Drug name → SMILES", "🧬 Paste SMILES"])

with tab1:
    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    name = st.text_input(
        "Enter drug / chemical name",
        value="ibuprofen",
        help="Use generic names like: aspirin, ibuprofen, acetaminophen/paracetamol, caffeine, metformin."
    )
    if st.button("Fetch SMILES"):
        s, src, err = name_to_smiles(name)
        if err:
            st.error(err)
        else:
            st.session_state.smiles = s
            st.session_state.smiles_source = src
            st.success(f"Saved SMILES via: {src}")
            st.code(s, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    s2 = st.text_area(
        "Paste SMILES here",
        value="",
        height=120,
        placeholder="Example (ibuprofen): CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    )
    if st.button("Use this SMILES"):
        if not s2.strip():
            st.error("Paste a SMILES string first.")
        else:
            s = max(s2.strip().split("."), key=len)
            st.session_state.smiles = s
            st.session_state.smiles_source = "Manual SMILES"
            st.success("Saved your SMILES")
            st.code(s, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

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
    if not st.session_state.smiles:
        st.error("First fetch SMILES by name OR paste a SMILES.")
        st.stop()

    smiles = st.session_state.smiles

    X = vec.transform([smiles])

    if X.nnz == 0:
        st.error(
            "Vectorizer found zero matching n-grams in this SMILES. "
            "This can happen if model + vectorizer files don’t match."
        )
        st.stop()

    # Correct class-1 probability
    classes = list(getattr(tox_model, "classes_", [0, 1]))
    if 1 not in classes or 0 not in classes:
        st.error(f"Model classes look wrong: {classes}. Retrain with labels 0/1.")
        st.stop()

    tox_proba_vec = tox_model.predict_proba(X)[0]
    p_tox = float(tox_proba_vec[classes.index(1)])
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

    # ---------- Results ----------
    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.subheader("Result")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("CT_TOX probability", f"{p_tox:.6f}")
    with c2:
        st.metric("Prediction", "TOXIC / POSITIVE" if toxic_flag else "NON-TOXIC / NEGATIVE")

    st.progress(int(round(p_tox * 100)))

    if p_fda is not None:
        st.caption(f"FDA_APPROVED probability (separate model): {p_fda:.4f}")

    st.caption(f"SMILES source: {st.session_state.smiles_source}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Explanation ----------
    st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
    st.subheader("Explanation (local)")
    st.write("Positive contribution pushes toward TOXIC; negative pushes toward NON-TOXIC.")
    st.dataframe(explain_local(smiles, tox_model, vec, topk=topk), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if show_debug:
        st.markdown('<div class="card fadeUp">', unsafe_allow_html=True)
        st.subheader("Debug")
        st.write("tox_model.classes_:", classes)
        st.write("X.nnz:", int(X.nnz))
        st.write("Threshold:", threshold)
        st.code(smiles)
        st.markdown("</div>", unsafe_allow_html=True)
