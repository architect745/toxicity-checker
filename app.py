import json
from urllib.parse import quote

import numpy as np
import requests
import streamlit as st
from joblib import load


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Drug Toxicity Checker (XAI)",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"


# =========================
# Styling (top nav spacing + nicer UI)
# =========================
st.markdown(
    """
<style>
/* Make whole app light and clean */
.stApp {
  background: radial-gradient(1200px 800px at 15% 10%, rgba(59,130,246,0.10), transparent 60%),
              radial-gradient(1000px 700px at 90% 10%, rgba(16,185,129,0.10), transparent 55%),
              #f8fafc;
  color: #0f172a;
}

/* Remove default sidebar nav if it appears */
[data-testid="stSidebarNav"] { display: none; }

/* Header styling */
.stAppHeader {
  background: rgba(248, 250, 252, 0.70) !important;
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(15, 23, 42, 0.08);
}

/* --- IMPORTANT PART: MORE SPACE IN TOP NAV --- */
/* Streamlit top nav uses rc-overflow (menu container) */
.stAppHeader .rc-overflow {
  gap: 18px !important;                 /* <-- big spacing between items */
  align-items: center !important;
}

/* Streamlit 1.46+ uses data-testid="stTopNavLink" for top links */
.stAppHeader [data-testid="stTopNavLink"] {
  padding: 10px 16px !important;        /* bigger clickable area */
  border-radius: 999px !important;
  font-weight: 600 !important;
  letter-spacing: 0.2px;
}

/* Hover effect */
.stAppHeader [data-testid="stTopNavLink"]:hover {
  background: rgba(59,130,246,0.12) !important;
}

/* Selected page highlight (best-effort; Streamlit changes internals sometimes) */
.stAppHeader [aria-current="page"] {
  background: rgba(59,130,246,0.18) !important;
  border: 1px solid rgba(59,130,246,0.25) !important;
}

/* Cards */
.card {
  background: rgba(255,255,255,0.85);
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
}

.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  border: 1px solid rgba(15, 23, 42, 0.12);
  background: rgba(255,255,255,0.8);
}

.pulse {
  position: relative;
}
.pulse::after {
  content: "";
  position: absolute;
  inset: -8px;
  border-radius: 999px;
  border: 2px solid rgba(59,130,246,0.25);
  animation: pulse 1.7s infinite;
  opacity: 0;
}
@keyframes pulse {
  0% { transform: scale(0.98); opacity: 0.0; }
  35% { opacity: 0.6; }
  100% { transform: scale(1.08); opacity: 0.0; }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Helpers
# =========================
def looks_like_smiles(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    # crude but works for beginners
    smiles_chars = set("=#[]()\\/+-@0123456789")
    return any(ch in smiles_chars for ch in t) and len(t) >= 6


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def pubchem_name_to_smiles(name: str) -> str | None:
    """
    Uses PubChem PUG REST to fetch IsomericSMILES from a compound name.
    """
    q = quote(name.strip())
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/property/IsomericSMILES/JSON"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None
    try:
        data = r.json()
        props = data["PropertyTable"]["Properties"]
        if not props:
            return None
        return props[0].get("IsomericSMILES") or None
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load(MODEL_PATH)
    vectorizer = load(VECTORIZER_PATH)
    return model, vectorizer


def predict_proba(smiles: str, model, vectorizer) -> float:
    X = vectorizer.transform([smiles])
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])
    # fallback for models without predict_proba
    pred = model.predict(X)[0]
    return float(pred)


def top_global_features(model, vectorizer, k=15):
    if not hasattr(model, "coef_"):
        return [], []
    coef = model.coef_.ravel()
    names = np.array(vectorizer.get_feature_names_out())
    top_pos = np.argsort(coef)[-k:][::-1]
    top_neg = np.argsort(coef)[:k]
    return list(zip(names[top_pos], coef[top_pos])), list(zip(names[top_neg], coef[top_neg]))


def local_contributions(smiles: str, model, vectorizer, k=10):
    if not hasattr(model, "coef_"):
        return []
    coef = model.coef_.ravel()
    X = vectorizer.transform([smiles])
    inds = X.indices
    vals = X.data
    names = vectorizer.get_feature_names_out()
    contrib = vals * coef[inds]
    pairs = [(names[i], float(c)) for i, c in zip(inds, contrib)]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:k]


# =========================
# Pages
# =========================
def page_home():
    st.markdown(
        """
<div class="card">
  <div class="badge">Mini Project • Explainable AI (XAI)</div>
  <h1 style="margin-top:10px;">Drug Toxicity Checker</h1>
  <p style="font-size:16px; line-height:1.55;">
    Type a <b>drug name</b> (like <i>ibuprofen</i>) or paste a <b>SMILES</b>.
    The app fetches SMILES from <b>PubChem</b> (if needed), runs your trained ML model,
    and shows an easy explanation using <b>n-gram contributions</b>.
  </p>
  <p style="opacity:0.8; margin-bottom:0;">
    ⚠️ Educational demo only — not medical advice.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown('<div class="card"><h3>1) Input</h3><p>Drug name or SMILES</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>2) Model</h3><p>SMILES n-grams → Logistic Regression</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h3>3) Explain</h3><p>Top n-grams pushing prediction up/down</p></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown(
        """
<div class="card">
<h3>What you can say to teachers (simple)</h3>
<ul>
  <li>We convert a drug to a text format called <b>SMILES</b>.</li>
  <li>We break SMILES into small text pieces (n-grams).</li>
  <li>A classifier predicts <b>Toxic vs Non-Toxic</b>.</li>
  <li>We show which n-grams influenced the decision (XAI).</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )


def page_predict():
    st.markdown('<div class="card"><h2 class="pulse">Predict Toxicity</h2><p>Enter a drug name or SMILES.</p></div>', unsafe_allow_html=True)
    st.write("")

    try:
        model, vectorizer = load_artifacts()
    except Exception as e:
        st.error(
            "Your app can't find/load model files.\n\n"
            "Make sure these exist in the repo root:\n"
            "- model.joblib\n"
            "- vectorizer.joblib\n\n"
            f"Details: {e}"
        )
        return

    q = st.text_input("Drug name or SMILES", placeholder="Example: ibuprofen OR CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    run = st.button("Predict", use_container_width=True)

    if not run:
        return

    if not q.strip():
        st.warning("Type something first.")
        return

    with st.spinner("Fetching SMILES + running model..."):
        if looks_like_smiles(q):
            smiles = q.strip()
            source = "You entered SMILES"
        else:
            smiles = pubchem_name_to_smiles(q)
            source = "PubChem (from name)"
            if smiles is None:
                st.error(
                    "PubChem returned no SMILES for that name.\n\n"
                    "Try simpler generic names like: aspirin, ibuprofen, metformin, paracetamol."
                )
                return

        p = predict_proba(smiles, model, vectorizer)

    # Interpret (you can change threshold if you want)
    label = "TOXIC" if p >= 0.5 else "NON-TOXIC"

    st.write("")
    st.markdown(
        f"""
<div class="card">
  <div class="badge">SMILES source: {source}</div>
  <h3 style="margin-top:10px;">Result: <span style="font-weight:800;">{label}</span></h3>
  <p style="font-size:16px;">Predicted probability (toxic): <b>{p:.3f}</b></p>
  <p style="margin-bottom:0;"><b>SMILES used:</b> <code>{smiles}</code></p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<div class="card"><h3>Quick Local Explanation</h3></div>', unsafe_allow_html=True)
    contribs = local_contributions(smiles, model, vectorizer, k=12)
    if not contribs:
        st.info("This model doesn’t expose coefficients, so local explanation is unavailable.")
        return

    st.write("Top n-grams with strongest influence (positive = more toxic, negative = less toxic):")
    st.dataframe(
        [{"n-gram": n, "contribution": c} for n, c in contribs],
        use_container_width=True,
        hide_index=True,
    )


def page_explain():
    st.markdown('<div class="card"><h2>Explain (XAI)</h2><p>Global: which patterns generally increase/decrease toxicity prediction.</p></div>', unsafe_allow_html=True)
    st.write("")

    try:
        model, vectorizer = load_artifacts()
    except Exception as e:
        st.error(
            "Your app can't find/load model files.\n\n"
            "Make sure these exist in the repo root:\n"
            "- model.joblib\n"
            "- vectorizer.joblib\n\n"
            f"Details: {e}"
        )
        return

    pos, neg = top_global_features(model, vectorizer, k=15)
    if not pos:
        st.info("This model doesn’t expose coef_, so global explanation is unavailable.")
        return

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="card"><h3>Increase toxicity prediction</h3></div>', unsafe_allow_html=True)
        st.dataframe(
            [{"n-gram": n, "weight": float(w)} for n, w in pos],
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.markdown('<div class="card"><h3>Decrease toxicity prediction</h3></div>', unsafe_allow_html=True)
        st.dataframe(
            [{"n-gram": n, "weight": float(w)} for n, w in neg],
            use_container_width=True,
            hide_index=True,
        )

    st.write("")
    st.markdown(
        """
<div class="card">
<h3>How to explain this to teachers (super simple)</h3>
<ul>
  <li>The model learned text patterns from SMILES.</li>
  <li><b>Positive weight</b> means that pattern pushes prediction towards “toxic”.</li>
  <li><b>Negative weight</b> means that pattern pushes prediction towards “non-toxic”.</li>
  <li>This is explainable because Logistic Regression has visible weights (coef).</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )


def page_about():
    st.markdown(
        """
<div class="card">
  <h2>About</h2>
  <p style="font-size:16px; line-height:1.6;">
    This mini project demonstrates an <b>Explainable AI</b> approach for drug analysis using simple machine learning.
    It predicts toxicity from chemical structure (SMILES) and explains the prediction using n-gram contributions.
  </p>
  <hr style="border:none; border-top:1px solid rgba(15,23,42,0.10);" />
  <p style="margin-bottom:6px;"><b>Developed by</b></p>
  <ul style="margin-top:0;">
    <li><b>Archit Jee</b></li>
    <li><b>Adarsh Singh</b></li>
  </ul>
  <p style="opacity:0.8; margin-bottom:0;">
    (Optional) You can add Semester/USN if your college expects it — it looks formal, but only add it if your teacher asked.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================
# Navigation (TOP)
# =========================
pages = [
    st.Page(page_home, title="Home"),
    st.Page(page_predict, title="Predict"),
    st.Page(page_explain, title="Explain"),
    st.Page(page_about, title="About"),
]

nav = st.navigation(pages, position="top")
nav.run()
