import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import quote

UTILS_VERSION = "utils_v2025_12_04_fix_pubchem_opsin"


# ---------- UI ----------
def inject_css():
    st.markdown(
        """
        <style>
          .card{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);
                border-radius:18px;padding:18px;margin:12px 0;}
          .muted{opacity:.85;font-size:.95rem}
          .fadeUp{animation:fadeUp .35s ease-out}
          @keyframes fadeUp{from{transform:translateY(8px);opacity:0}to{transform:translateY(0);opacity:1}}
          .pill{display:inline-block;padding:6px 10px;border-radius:999px;
                border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.06);
                font-size:.85rem;margin-right:8px;margin-top:6px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- Loading ----------
@st.cache_resource
def load_artifacts():
    return joblib.load("clintox_artifacts.joblib")


def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# ---------- Name cleaning (helps if you type brand names) ----------
_BRAND_MAP = {
    "dolo 650": "paracetamol",
    "crocin": "paracetamol",
    "brufen": "ibuprofen",
    "disprin": "aspirin",
    "glycomet": "metformin",
}

def _normalize_name(name: str) -> str:
    q = (name or "").strip().lower()
    q = " ".join(q.split())
    return _BRAND_MAP.get(q, q)


# ---------- PubChem + OPSIN ----------
def _keep_biggest_fragment(smiles: str) -> str:
    # removes salts/mixtures like ".Na"
    return max(str(smiles).split("."), key=len)

def _pubchem_get_cids(q: str):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(q)}/cids/JSON"
    r = requests.get(url, timeout=12)
    if r.status_code != 200:
        return []
    data = r.json()
    return data.get("IdentifierList", {}).get("CID", []) or []

def _pubchem_cid_to_smiles(cid: int):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES,CanonicalSMILES/JSON"
    r = requests.get(url, timeout=12)
    if r.status_code != 200:
        return None
    data = r.json()
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return None
    p = props[0]
    s = p.get("IsomericSMILES") or p.get("CanonicalSMILES")
    if not s:
        return None
    return _keep_biggest_fragment(s)

def _opsin_name_to_smiles(q: str):
    # OPSIN API: https://opsin.ch.cam.ac.uk/opsin/<name>.json :contentReference[oaicite:1]{index=1}
    url = f"https://opsin.ch.cam.ac.uk/opsin/{quote(q)}.json"
    r = requests.get(url, timeout=12)
    if r.status_code != 200:
        return None
    data = r.json()
    # OPSIN returns "smiles" when it works
    s = data.get("smiles")
    if not s:
        return None
    return _keep_biggest_fragment(s)

def name_to_smiles(name: str, max_cids_try: int = 10):
    """
    Returns (smiles, source_label, error)
    source_label = "PubChem CID 123" or "OPSIN"
    """
    raw = (name or "").strip()
    if not raw:
        return None, None, "Enter a drug/chemical name."

    q = _normalize_name(raw)

    # Try PubChem: name -> CID list -> try multiple CIDs until one gives SMILES :contentReference[oaicite:2]{index=2}
    try:
        cids = _pubchem_get_cids(q)
        for cid in cids[:max_cids_try]:
            s = _pubchem_cid_to_smiles(cid)
            if s:
                return s, f"PubChem CID {cid}", None
    except Exception:
        pass

    # Fallback: OPSIN (works for many chemical/IUPAC-ish/common names) :contentReference[oaicite:3]{index=3}
    try:
        s2 = _opsin_name_to_smiles(q)
        if s2:
            return s2, "OPSIN", None
    except Exception:
        pass

    return None, None, "Could not get SMILES. Try a generic/chemical name (aspirin, ibuprofen, caffeine, metformin) or paste SMILES manually."


# ---------- Explainability helpers ----------
def _safe_class1_coef(model):
    coefs = model.coef_.ravel().copy()
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if len(classes) == 2 and classes.index(1) != 1:
            coefs = -coefs
    return coefs

def explain_local(smiles: str, model, vectorizer, topk: int = 12):
    X = vectorizer.transform([smiles])
    feat_names = vectorizer.get_feature_names_out()
    coefs = _safe_class1_coef(model)

    idx = X.nonzero()[1]
    vals = X.data
    contrib = vals * coefs[idx]

    df = pd.DataFrame({"ngram": feat_names[idx], "contribution": contrib})
    if df.empty:
        return pd.DataFrame({"ngram": [], "contribution": []})

    df = df.sort_values("contribution", ascending=False)
    out = pd.concat([df.head(topk), df.tail(topk).sort_values("contribution")], axis=0)
    out["contribution"] = out["contribution"].round(6)
    return out.reset_index(drop=True)

def top_global_ngrams(model, vectorizer, topk: int = 20):
    feat_names = vectorizer.get_feature_names_out()
    coefs = _safe_class1_coef(model)
    idx_sorted = np.argsort(coefs)
    top_neg = [(feat_names[i], float(coefs[i])) for i in idx_sorted[:topk]]
    top_pos = [(feat_names[i], float(coefs[i])) for i in idx_sorted[-topk:][::-1]]
    df_pos = pd.DataFrame(top_pos, columns=["ngram", "weight"]).round(6)
    df_neg = pd.DataFrame(top_neg, columns=["ngram", "weight"]).round(6)
    return df_pos, df_neg
