import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

BG_URL = "https://files.catbox.moe/evxdoq.png"

def inject_css(bg_url: str = BG_URL):
    css = f"""
    <style>
      .stApp {{
        background: url("{bg_url}") no-repeat center center fixed;
        background-size: cover;
      }}

      /* subtle dark overlay for readability */
      .block-container {{
        background: rgba(0,0,0,0.45);
        border-radius: 18px;
        padding: 22px 22px 28px 22px;
      }}

      .card {{
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 18px;
        padding: 16px 16px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.25);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        margin-bottom: 14px;
      }}

      .muted {{
        opacity: 0.85;
        font-size: 0.95rem;
      }}

      .fadeUp {{
        animation: fadeUp 0.5s ease-out;
      }}

      @keyframes fadeUp {{
        from {{ transform: translateY(8px); opacity: 0; }}
        to   {{ transform: translateY(0px); opacity: 1; }}
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    base = Path("artifacts")
    model = joblib.load(base / "model.joblib")
    vec = joblib.load(base / "vectorizer.joblib")

    label_path = base / "label.txt"
    label_name = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else "TOXIC"
    return model, vec, label_name

def predict_smiles(smiles: str, model, vec) -> float:
    X = vec.transform([smiles])

    # Prefer proper probabilities if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # assume positive class is index 1
        return float(proba[0, 1])

    # fallback: convert decision_function -> pseudo probability
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    # last resort
    pred = model.predict(X)[0]
    return float(pred)

def top_global_ngrams(model, vec, k: int = 15):
    if not hasattr(model, "coef_"):
        raise ValueError("Your model has no coef_. Use a linear model (LogReg / LinearSVC) to get global n-gram weights.")

    coef = model.coef_
    # binary case: (1, n_features)
    weights = coef[0] if coef.ndim == 2 else coef

    features = vec.get_feature_names_out()

    top_pos_idx = np.argsort(weights)[-k:][::-1]
    top_neg_idx = np.argsort(weights)[:k]

    pos = pd.DataFrame({
        "pattern": features[top_pos_idx],
        "weight": weights[top_pos_idx]
    })

    neg = pd.DataFrame({
        "pattern": features[top_neg_idx],
        "weight": weights[top_neg_idx]
    })

    return pos, neg
