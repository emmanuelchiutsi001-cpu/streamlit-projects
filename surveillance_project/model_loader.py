# model_loader.py
import torch
import pickle
import streamlit as st
import os


@st.cache_resource
def load_pretrained_model(model_path="saved_model.pkl"):
    """Load pre-trained model from file"""
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Import here to avoid circular imports
        from train_model import OptimizedCrimeClassifier

        # Create model instance
        model = OptimizedCrimeClassifier(num_classes=model_data['num_classes'])
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()

        return {
            'model': model,
            'class_mapping': model_data['class_mapping'],
            'inverse_class_mapping': model_data['inverse_class_mapping'],
            'effective_class_names': model_data['effective_class_names'],
            'performance_metrics': model_data['performance_metrics'],
            'training_info': model_data['training_info'],
            'training_history': model_data['training_history']
        }
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None