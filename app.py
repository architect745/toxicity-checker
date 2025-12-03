import time
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

# ---------- PubChem: name -> SMILES (more robust) ----------
HEADERS = {"User-Agent": "toxicity-checker/1.0 (streamlit app)"}

def pubchem_name_to_smiles(name: str):
    name = name.strip()
    if not name:
        return None, None, "Please type a drug/compound name."

    q = quote(name)

    # 1) name -> CID list (JSON)
    cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/cids/JSON"
    r1 = requests.get(cid_url, headers=HEADERS, timeout=20)
    if r1.status_code != 200:
        return None, None, f"PubChem CID lookup failed (HTTP {r1.status_code}). Try again."

    try:
        j1 = r1.json()
    except Exception:
        return None, None, "PubChem CID response was not JSON (possible throttling). Try again later."

    # PubChem error payload
    if "Fault" in j1:
        msg = j1["Fault"].get("Message", "PubChem error")
        details = j1["Fault"].get("Details", [])
        more = details[0] if details else ""
        return None, None, f"{msg}. {more}".strip()

    cids = j1.get("IdentifierList", {}).get("CID", [])
    if not cids:
        return None, None, "No CID found in PubChem for this name. Try a different name."

    # 2) try several CIDs until one gives a real SMILES (TXT)
    for cid in cids[:10]:
        smi_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT"
        r2 = requests.get(smi_url, headers=HEADERS, timeout=20)

        if r2.status_code == 200:
            smiles = r2.text.strip()
            # sanity checks: empty or HTML is not SMILES
            if smiles and "<html" not in smiles.lower() and "<!doctype" not in smiles.lower():
                return smiles, cid, None

        time.sleep(0.2)  # gentle throttle

    return None, cids[0], "PubChem matched the name, but SMILES could not be retrieved (maybe throttled). Try again later."


# ---------- Explanation (local contributions from Logistic Regression) ----------
def explain(smiles: str, topk=12):
    x = vec.transform([smiles])
    w = model.coef_[0]
    feats = np.array(vec.get_feature_names_out())

    if x.nnz == 0:
        return []

    idxs = x.indices
    vals = x.data
    contrib = vals * w[idxs]  # approximate local contribution

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
st.write(f"Model task: **{label_name}** (Tox21 assay label). Demo prediction only — not medical advice.")

drug_name = st.text_input("Enter drug/chemical name", value="ibuprofen")
threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05)
topk = st.slider("How many explanation patterns to show", 5, 20, 12, 1)

if st.button("Predict toxicity"):
    smiles, cid, err = pubchem_name_to_smiles(drug_name)
    if err:
        st.error(err)
        if cid is not None:
            st.write("Matched CID (first):", cid)
        st.info("Try: caffeine, aspirin, ibuprofen, metformin, acetaminophen (paracetamol).")
        st.stop()

    st.success(f"PubChem CID: {cid}")
    st.code(smiles, language="text")

    try:
        X = vec.transform([smiles])
        p = float(model.predict_proba(X)[:, 1][0])
    except Exception as e:
        st.error("Model prediction failed.")
        st.exception(e)
        st.stop()

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
