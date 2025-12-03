import streamlit as st
import requests
import numpy as np
import joblib

st.set_page_config(page_title="Toxicity Checker (Demo)", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("tox_model.joblib")
    vec = joblib.load("tox_vectorizer.joblib")
    label = open("tox_label.txt").read().strip()
    return model, vec, label

model, vec, label_name = load_artifacts()

st.title("Drug Toxicity Checker (Demo)")
st.write(f"This predicts **{label_name} assay activity** from Tox21 (demo only).")

drug_name = st.text_input("Enter drug/chemical name", value="caffeine")

def pubchem_name_to_smiles(name: str):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return None, f"PubChem lookup failed (status {r.status_code}). Try another name."
    try:
        js = r.json()
        smiles = js["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        return smiles, None
    except Exception:
        return None, "Could not parse PubChem response. Try another name."

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
            "ngram": feats[idxs[i]],
            "count": int(vals[i]),
            "weight": float(w[idxs[i]]),
            "contribution": float(contrib[i])
        })
    return rows

if st.button("Predict toxicity"):
    smiles, err = pubchem_name_to_smiles(drug_name.strip())
    if err:
        st.error(err)
    else:
        st.success(f"Found SMILES: {smiles}")

        X = vec.transform([smiles])
        p = float(model.predict_proba(X)[:, 1][0])

        st.metric("Predicted probability (toxic in this assay)", f"{p:.3f}")

        st.write("Prediction:", "**TOXIC**" if p >= 0.5 else "**NON-TOXIC**")

        st.subheader("Explanation (top contributing SMILES patterns)")
        st.write("Positive contribution pushes towards TOXIC. Negative pushes towards NON-TOXIC.")
        rows = explain(smiles, topk=12)
        st.dataframe(rows, use_container_width=True)
