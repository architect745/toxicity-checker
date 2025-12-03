import streamlit as st
import requests
import numpy as np
import joblib
from urllib.parse import quote

# ---------- Page config (must be before any other st.* calls) ----------
st.set_page_config(page_title="Toxicity Checker (Demo)", page_icon="🧪", layout="centered")

# ---------- Load model artifacts safely ----------
try:
    model = joblib.load("tox_model.joblib")
    vec = joblib.load("tox_vectorizer.joblib")
    label_name = open("tox_label.txt").read().strip()
except Exception as e:
    st.error("App failed to start (could not load model files).")
    st.exception(e)
    st.stop()

# ---------- PubChem: name -> SMILES (robust) ----------
def pubchem_name_to_smiles(name: str):
    name = name.strip()
    if not name:
        return None, "Please enter a compound name."

    q = quote(name)

    # Step 1: name -> CID
    cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/cids/JSON"
    r1 = requests.get(cid_url, timeout=20)

    try:
        j1 = r1.json()
    except Exception:
        return None, f"PubChem did not return JSON (CID request). Status={r1.status_code}. Response: {r1.text[:200]}"

    if "Fault" in j1:
        msg = j1["Fault"].get("Message", "PubChem error")
        details = j1["Fault"].get("Details", [])
        more = details[0] if details else ""
        return None, f"{msg}. {more}".strip()

    cids = j1.get("IdentifierList", {}).get("CID", [])
    if not cids:
        return None, "Compound not found in PubChem. Try a different name (generic/chemical name)."

    cid = cids[0]

    # Step 2: CID -> Canonical SMILES
    smi_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    r2 = requests.get(smi_url, timeout=20)

    try:
        j2 = r2.json()
    except Exception:
        return None, f"PubChem did not return JSON (SMILES request). Status={r2.status_code}. Response: {r2.text[:200]}"

    if "Fault" in j2:
        msg = j2["Fault"].get("Message", "PubChem error")
        details = j2["Fault"].get("Details", [])
        more = details[0] if details else ""
        return None, f"{msg}. {more}".strip()

    props = j2.get("PropertyTable", {}).get("Properties", [])
    if not props or "CanonicalSMILES" not in props[0]:
        return None, "PubChem returned no SMILES for this compound."

    return props[0]["CanonicalSMILES"], None


# ---------- Explanation (local) ----------
def explain(smiles: str, topk=12):
    x = vec.transform([smiles])
    w = model.coef_[0]
    feats = np.array(vec.get_feature_names_out())

    if x.nnz == 0:
        return []

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


# ---------- UI ----------
st.title("🧪 Drug Toxicity Checker (Demo)")
st.write(f"Model task: **{label_name}** (Tox21 assay). This is a demo prediction, not medical advice.")

drug_name = st.text_input("Enter drug/chemical name", value="caffeine")
threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
topk = st.slider("How many explanation patterns to show", 5, 20, 12, 1)

if st.button("Predict toxicity"):
    smiles, err = pubchem_name_to_smiles(drug_name)
    if err:
        st.error(err)
        st.info("Tip: Try a generic/chemical name (e.g., aspirin, ibuprofen, metformin, paracetamol).")
        st.stop()

    st.success("Found compound in PubChem.")
    st.code(smiles, language="text")

    X = vec.transform([smiles])
    p = float(model.predict_proba(X)[:, 1][0])

    st.metric("Predicted probability (positive/toxic in this assay)", f"{p:.3f}")

    pred_label = "TOXIC / POSITIVE" if p >= threshold else "NON-TOXIC / NEGATIVE"
    st.write("Prediction:", f"**{pred_label}**")

    st.progress(int(round(p * 100)))

    st.subheader("Explanation (top SMILES patterns)")
    st.write("Positive contribution pushes towards TOXIC. Negative pushes towards NON-TOXIC.")
    rows = explain(smiles, topk=topk)
    if not rows:
        st.warning("No matching patterns found for this SMILES (rare). Try another compound.")
    else:
        st.dataframe(rows, use_container_width=True)
