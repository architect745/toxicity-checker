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
    return joblib.load("clintox_artifacts.joblib")


def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# ---------- PubChem ----------
def _extract_smiles_from_properties(props: dict):
    # Prefer IsomericSMILES if present, else CanonicalSMILES
    s = props.get("IsomericSMILES") or props.get("CanonicalSMILES")
    if not s:
        return None
    # remove salts/mixtures: keep largest fragment
    return max(str(s).split("."), key=len)


def pubchem_name_to_smiles(name: str):
    """
    name -> (SMILES, CID, error)
    Tries:
      1) name -> property(IsomericSMILES,CanonicalSMILES,CID)
      2) name -> CID list -> property(IsomericSMILES,CanonicalSMILES)
    """
    q = (name or "").strip()
    if not q:
        return None, None, "Enter a drug/chemical name."

    q_enc = quote(q)

    # 1) Direct fetch (try both SMILES types)
    url1 = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q_enc}"
        f"/property/IsomericSMILES,CanonicalSMILES,CID/JSON"
    )
    try:
        r1 = requests.get(url1, timeout=12)
        if r1.status_code == 200:
            data = r1.json()
            # PubChem sometimes returns Fault JSON
            if "Fault" in data:
                return None, None, data["Fault"].get("Message", "PubChem error. Try another name.")

            props_list = data.get("PropertyTable", {}).get("Properties", [])
            # If multiple matches, pick first that actually has SMILES
            for props in props_list:
                cid = props.get("CID")
                smiles = _extract_smiles_from_properties(props)
                if smiles:
                    return smiles, cid, None
            # If we got here, no entry in the list had SMILES
            # but there may still be a CID
            cid0 = props_list[0].get("CID") if props_list else None
            return None, cid0, "PubChem returned no SMILES for this compound."
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

        # 3) CID -> SMILES (try both SMILES types)
        url3 = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}"
            f"/property/IsomericSMILES,CanonicalSMILES/JSON"
        )
        r3 = requests.get(url3, timeout=12)
        if r3.status_code != 200:
            return None, cid, "Found a CID but couldn't fetch SMILES. Try another name."

        data3 = r3.json()
        if "Fault" in data3:
            return None, cid, data3["Fault"].get("Message", "PubChem error. Try another name.")

        props = data3["PropertyTable"]["Properties"][0]
        smiles = _extract_smiles_from_properties(props)
        if not smiles:
            return None, cid, "PubChem returned no SMILES for this compound."
        return smiles, cid, None

    except Exception:
        return None, None, "Could not parse PubChem response. Try another name."


# ---------- Explainability ----------
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
    top_pos = df.head(topk)
    top_neg = df.tail(topk).sort_values("contribution", ascending=True)

    out = pd.concat([top_pos, top_neg], axis=0)
    out["contribution"] = out["contribution"].round(6)
    return out.reset_index(drop=True)


def top_global_ngrams(model, vectorizer, topk: int = 20):
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
