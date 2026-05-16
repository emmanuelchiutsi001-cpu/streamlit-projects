# test_local.py
import streamlit as st
from model_loader import load_pretrained_model

st.title("Model Test")
model_data = load_pretrained_model("saved_model.pkl")

if model_data:
    st.success("✅ Model loaded successfully!")
    st.write(f"Classes: {model_data['effective_class_names']}")
    st.write(f"Accuracy: {model_data['performance_metrics']['accuracy']:.1f}%")
else:
    st.error("Failed to load model")