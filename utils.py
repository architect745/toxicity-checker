import joblib, requests
import pandas as pd
import streamlit as st
from urllib.parse import quote

def inject_css():
    st.markdown("""
    <style>
      .card{background:#ffffff12;border:1px solid #ffffff22;border-radius:18px;padding:18px;margin:12px 0;}
      .muted{opacity:.8;font-size:.95rem}
      .fadeUp{animation:fadeUp .35s ease-out}
      @keyframes fadeUp{from{transform:translateY(8px);opacity:0}to{transform:translateY(0);opacity:1}}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    a = joblib.load("clintox_artifacts.joblib")
    return a

def pubchem_name_to_smiles(name: str):
    q = (name or "").strip()
    if not q:
        return None, None, "Enter a drug/chemical name."

    # PubChem PUG REST: compound/name/<name>/property/CanonicalSMILES/JSON :contentReference[oaicite:1]{index=1}
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
        # remove salts/mixtures: keep biggest fragment
        smiles = max(smiles.split("."), key=len)
        return smiles, cid, None
    except Exception:
        return None, None, "Could not parse PubChem response. Try another name."

def explain_local(smiles: str, model, vectorizer, topk: int = 12):
    X = vectorizer.transform([smiles])
    feat_names = vectorizer.get_feature_names_out()
    coefs = model.coef_.ravel()

    # safety: make coef direction correspond to class 1
    if hasattr(model, "classes_") and len(model.classes_) == 2 and list(model.classes_)[1] != 1:
        coefs = -coefs

    idx = X.nonzero()[1]
    vals = X.data
    contrib = vals * coefs[idx]

    df = pd.DataFrame({
        "ngram": feat_names[idx],
        "contribution": contrib
    }).sort_values("contribution", ascending=False)

    top_pos = df.head(topk)
    top_neg = df.tail(topk).sort_values("contribution", ascending=True)

    out = pd.concat([top_pos, top_neg], axis=0)
    out["contribution"] = out["contribution"].round(6)
    return out.reset_index(drop=True)
