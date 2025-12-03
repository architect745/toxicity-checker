import time
import requests
import numpy as np
import joblib
import streamlit as st
from urllib.parse import quote

HEADERS = {"User-Agent": "toxicity-checker/1.0 (streamlit app)"}

@st.cache_resource
def load_artifacts():
    model = joblib.load("tox_model.joblib")
    vec = joblib.load("tox_vectorizer.joblib")
    label_name = open("tox_label.txt").read().strip()
    return model, vec, label_name

def pubchem_name_to_smiles(name: str):
    name = name.strip()
    if not name:
        return None, None, "Please type a drug/compound name."

    q = quote(name)

    # name -> CID list
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

    # try several CIDs -> SMILES (TXT)
    for cid in cids[:10]:
        smi_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT"
        r2 = requests.get(smi_url, headers=HEADERS, timeout=20)

        if r2.status_code == 200:
            smiles = r2.text.strip()
            if smiles and "<html" not in smiles.lower() and "<!doctype" not in smiles.lower():
                return smiles, cid, None

        time.sleep(0.2)

    return None, cids[0], "PubChem matched the name, but SMILES could not be retrieved. Try again later."

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

def top_global_ngrams(model, vec, k=12):
    w = model.coef_[0]
    feats = np.array(vec.get_feature_names_out())
    pos_idx = np.argsort(w)[::-1][:k]
    neg_idx = np.argsort(w)[:k]
    pos = [{"pattern": feats[i], "weight": float(w[i])} for i in pos_idx]
    neg = [{"pattern": feats[i], "weight": float(w[i])} for i in neg_idx]
    return pos, neg

def inject_css():
    st.markdown("""
    <style>
      /* Page background gradient */
      [data-testid="stAppViewContainer"]{
        background: radial-gradient(1200px circle at 10% 10%, rgba(108,99,255,.20), transparent 45%),
                    radial-gradient(1200px circle at 90% 20%, rgba(0,200,255,.14), transparent 40%),
                    linear-gradient(180deg, #f7f7ff 0%, #f6f7fb 60%, #f2f4ff 100%);
      }

      /* Center width + spacing */
      .block-container {max-width: 1050px; padding-top: 1.4rem; padding-bottom: 2rem;}

      /* Card */
      .card{
        background: rgba(255,255,255,.75);
        border: 1px solid rgba(15,23,42,.10);
        border-radius: 18px;
        padding: 18px 18px;
        box-shadow: 0 10px 30px rgba(17,24,39,.08);
        backdrop-filter: blur(8px);
        margin-bottom: 14px;
      }

      /* Fade-in animation */
      @keyframes fadeUp {
        from {opacity: 0; transform: translateY(10px);}
        to   {opacity: 1; transform: translateY(0);}
      }
      .fadeUp{ animation: fadeUp .5s ease-out both; }

      /* Button styling */
      .stButton button{
        border-radius: 999px !important;
        padding: .55rem 1rem !important;
        border: 1px solid rgba(15,23,42,.12) !important;
        box-shadow: 0 10px 25px rgba(108,99,255,.12) !important;
      }
      .stButton button:hover{
        transform: translateY(-1px);
        transition: 0.15s ease;
      }

      /* Hide Streamlit footer */
      footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

