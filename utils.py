import time
import requests
import numpy as np
import joblib
import streamlit as st
from urllib.parse import quote

HEADERS = {"User-Agent": "toxicity-checker/1.0 (streamlit app)"}

# ---------- Styling / animation ----------
def inject_css():
    st.markdown("""
    <style>
      /* Background */
      [data-testid="stAppViewContainer"]{
        background: radial-gradient(1200px circle at 10% 10%, rgba(108,99,255,.22), transparent 45%),
                    radial-gradient(1200px circle at 90% 20%, rgba(0,200,255,.16), transparent 40%),
                    linear-gradient(180deg, #f7f7ff 0%, #f6f7fb 60%, #f2f4ff 100%);
      }
      .block-container {max-width: 1050px; padding-top: 1.3rem; padding-bottom: 2rem;}

      /* Sidebar tweaks */
      [data-testid="stSidebar"]{
        background: rgba(255,255,255,.70);
        border-right: 1px solid rgba(15,23,42,.10);
        backdrop-filter: blur(10px);
      }

      /* Animations */
      @keyframes fadeUp { from {opacity:0; transform: translateY(10px);} to {opacity:1; transform: translateY(0);} }
      @keyframes floaty { 0% {transform: translateY(0);} 50% {transform: translateY(-4px);} 100% {transform: translateY(0);} }
      .fadeUp { animation: fadeUp .45s ease-out both; }
      .floaty { animation: floaty 2.6s ease-in-out infinite; }

      /* Cards */
      .card{
        background: rgba(255,255,255,.78);
        border: 1px solid rgba(15,23,42,.10);
        border-radius: 18px;
        padding: 18px 18px;
        box-shadow: 0 14px 34px rgba(17,24,39,.10);
        backdrop-filter: blur(10px);
        margin-bottom: 14px;
      }
      .hero-title{
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 .2rem 0;
        letter-spacing: -0.02em;
      }
      .muted{opacity:.75; font-size: .98rem;}
      .badge{
        display:inline-block;
        padding: .25rem .65rem;
        border-radius: 999px;
        border: 1px solid rgba(15,23,42,.14);
        font-size: .85rem;
        margin-right: .35rem;
        background: rgba(255,255,255,.55);
      }

      /* Buttons */
      .stButton button{
        border-radius: 999px !important;
        padding: .55rem 1.05rem !important;
        border: 1px solid rgba(15,23,42,.12) !important;
        box-shadow: 0 12px 28px rgba(108,99,255,.14) !important;
      }
      .stButton button:hover{
        transform: translateY(-1px);
        transition: 0.15s ease;
      }

      /* Remove footer */
      footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ---------- Load artifacts (cached) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("tox_model.joblib")
    vec = joblib.load("tox_vectorizer.joblib")
    label_name = open("tox_label.txt").read().strip()
    return model, vec, label_name

# ---------- Lottie loader ----------
@st.cache_data(show_spinner=False)
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=20, headers=HEADERS)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# ---------- PubChem name -> SMILES ----------
def pubchem_name_to_smiles(name: str):
    name = name.strip()
    if not name:
        return None, None, "Please type a drug/compound name."

    q = quote(name)

    cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/cids/JSON"
    r1 = requests.get(cid_url, headers=HEADERS, timeout=20)
    if r1.status_code != 200:
        return None, None, f"PubChem CID lookup failed (HTTP {r1.status_code}). Try again."

    try:
        j1 = r1.json()
    except Exception:
        return None, None, "PubChem CID response was not JSON (possible throttling). Try again later."

    if "Fault" in j1:
        msg = j1["Fault"].get("Message", "PubChem error")
        details = j1["Fault"].get("Details", [])
        more = details[0] if details else ""
        return None, None, f"{msg}. {more}".strip()

    cids = j1.get("IdentifierList", {}).get("CID", [])
    if not cids:
        return None, None, "No CID found in PubChem for this name. Try a generic/chemical name."

    for cid in cids[:10]:
        smi_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT"
        r2 = requests.get(smi_url, headers=HEADERS, timeout=20)
        if r2.status_code == 200:
            smiles = r2.text.strip()
            if smiles and "<html" not in smiles.lower() and "<!doctype" not in smiles.lower():
                return smiles, cid, None
        time.sleep(0.2)

    return None, cids[0], "PubChem matched the name, but SMILES could not be retrieved. Try again later."

# ---------- Explainability helpers ----------
def explain_local(smiles: str, model, vec, topk=12):
    x = vec.transform([smiles])
    if x.nnz == 0:
        return []

    w = model.coef_[0]
    feats = np.array(vec.get_feature_names_out())

    idxs = x.indices
    vals = x.data
    contrib = vals * w[idxs]

    order = np.argsort(np.abs(contrib))[::-1][:topk]
    rows = []
    for i in order:
        rows.append({
            "pattern": feats[idxs[i]],
            "count": int(vals[i]),
            "weight": float(w[idxs[i]]),
            "contribution": float(contrib[i])
        })
    return rows

def top_global_ngrams(model, vec, k=15):
    w = model.coef_[0]
    feats = np.array(vec.get_feature_names_out())
    pos_idx = np.argsort(w)[::-1][:k]
    neg_idx = np.argsort(w)[:k]
    pos = [{"pattern": feats[i], "weight": float(w[i])} for i in pos_idx]
    neg = [{"pattern": feats[i], "weight": float(w[i])} for i in neg_idx]
    return pos, neg
