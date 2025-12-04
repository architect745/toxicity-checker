import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import quote


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
    """
    Expects: clintox_artifacts.joblib in repo root.
    That file should contain:
      vectorizer, tox_model, fda_model, best_threshold, label_note
    """
    return joblib.load("clintox_artifacts.joblib")


def load_lottie_url(url: str):
    """Fetch a lottie JSON safely."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# ---------- PubChem helpers ----------
def pubchem_name_to_smiles(name: str):
    """
    Convert a drug/compound name to Canonical SMILES using PubChem PUG REST.
    Includes fallback (name->CID list -> SMILES).
    Returns: (smiles, cid, error_message)
    """
    q = (name or "").strip()
    if not q:
        return None, None, "Enter a drug/chemical name."

    q_enc = quote(q)

    # 1) Direct: name -> SMILES + CID
    url1 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q_enc}/property/CanonicalSMILES,CID/JSON"
    try:
        r1 = requests.get(url1, timeout=12)
        if r1.status_code == 200:
            data = r1.json()
            props = data["PropertyTable"]["Properties"][0]
            smiles = props.get("CanonicalSMILES")
            cid = props.get("CID")
            if smiles:
                # remove salts/mixtures: keep biggest fragment
                smiles = max(str(smiles).split("."), key=len)
                return smiles, cid, None
    except Exception:
        pass

    # 2) Fallback: name -> CID(s)
    url2 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q_enc}/cids/JSON"
    try:
        r2 = requests.get(url2, timeout=12)
        if r2.status_code != 200:
            return None, None, "PubChem couldn't find that name. Try a generic name (aspirin, ibuprofen, metformin)."

        cid_list = r2.json().get("IdentifierList", {}).get("CID", [])
        if not cid_list:
            return None, None, "PubChem returned no CID for this name. Try another name."

        cid = cid_list[0]

        # 3) CID -> SMILES
        url3 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
        r3 = requests.get(url3, timeout=12)
        if r3.status_code != 200:
            return None, cid, "Found a CID but couldn't fetch SMILES. Try another name."

        props = r3.json()["PropertyTable"]["Properties"][0]
        smiles = props.get("CanonicalSMILES")
        if not smiles:
            return None, cid, "PubChem returned no SMILES for this compound."

        smiles = max(str(smiles).split("."), key=len)
        return smiles, cid, None

    except Exception:
        return None, None, "Could not parse PubChem response. Try another name."


# ---------- Explainability ----------
def _safe_class1_coef(model):
    """
    Returns coefficients aligned so positive means pushing toward class 1.
    Works for binary linear classifiers (LogisticRegression).
    """
    coefs = model.coef_.ravel().copy()
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        # if class 1 is not at index 1, swap direction
        if len(classes) == 2 and classes.index(1) != 1:
            coefs = -coefs
    return coefs


def explain_local(smiles: str, model, vectorizer, topk: int = 12):
    """
    Local explanation: contribution = tfidf_value * coef for features present in SMILES.
    Returns dataframe with top positive and top negative contributions.
    """
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

    top_pos = df.head(topk)
    top_neg = df.tail(topk).sort_values("contribution", ascending=True)

    out = pd.concat([top_pos, top_neg], axis=0)
    out["contribution"] = out["contribution"].round(6)
    return out.reset_index(drop=True)


def top_global_ngrams(model, vectorizer, topk: int = 20):
    """
    Global explanation: show n-grams with largest positive and negative weights.
    """
    feat_names = vectorizer.get_feature_names_out()
    coefs = _safe_class1_coef(model)

    idx_sorted = np.argsort(coefs)
    top_neg = [(feat_names[i], float(coefs[i])) for i in idx_sorted[:topk]]
    top_pos = [(feat_names[i], float(coefs[i])) for i in idx_sorted[-topk:][::-1]]

    df_pos = pd.DataFrame(top_pos, columns=["ngram", "weight"])
    df_neg = pd.DataFrame(top_neg, columns=["ngram", "weight"])
    df_pos["weight"] = df_pos["weight"].round(6)
    df_neg["weight"] = df_neg["weight"].round(6)
    return df_pos, df_neg
