import streamlit as st
import os
import joblib

st.set_page_config(page_title="Debug", layout="centered")
st.title("Debug page")

st.write("Files in this app folder:")
st.write(os.listdir("."))

try:
    model = joblib.load("tox_model.joblib")
    vec = joblib.load("tox_vectorizer.joblib")
    label = open("tox_label.txt").read().strip()
    st.success("✅ Model files loaded OK")
    st.write("Label:", label)
except Exception as e:
    st.error("❌ Crash while loading model files")
    st.exception(e)
    st.stop()

st.success("✅ Streamlit is running app.py correctly.")
