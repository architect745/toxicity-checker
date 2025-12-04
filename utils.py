import joblib
import requests
import pandas as pd
import streamlit as st
from urllib.parse import quote

def inject_css():
    st.markdown(
        """
        <style>
          .card{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);
                border-radius:18px;padding:18px;margin:12px 0;}
          .muted{opacity:.85;font-size:.95rem}
        </style>
        """,
        unsafe_allow_html=True,
    )

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

def pubchem_name_to_smiles(name: str):
    q = (name or "").strip()
    if not q:
        return None, None, "Enter a drug/chemical name."

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(q)}/property/CanonicalSMILES,CID/JSON"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return None, None, "PubChem couldn't find that name. Try another (aspirin, ibuprofen, metformin)."
        data = r.json()
        props = data["PropertyTable"]["Properties"][0]
        smiles = props.get("CanonicalSMILES")
        cid = props.get("CID")
        if not smiles:
            return None, cid, "PubChem returned no SMILES for this compound."
        smiles = max(smiles.split("."), key=len)
        return smiles, cid, None
    except Exception:
        return None, None, "Could not parse PubChem response. Try another name."

def explain_local(smiles: str, model, vectorizer, topk: int = 12):
    X = vectorizer.transform([smiles])
    feat_names = vectorizer.get_feature_names_out()
    coefs = model.coef_.ravel()

    idx = X.nonzero()[1]
    vals = X.data
    contrib = vals * coefs[idx]

    df = pd.DataFrame({"ngram": feat_names[idx], "contribution": contrib})
    df = df.sort_values("contribution", ascending=False)

    top_pos = df.head(topk)
    top_neg = df.tail(topk).sort_values("contribution", ascending=True)

    out = pd.concat([top_pos, top_neg], axis=0)
    out["contribution"] = out["contribution"].round(6)
    return out.reset_index(drop=True)
