# ==============================
# Prediction Module (Improved)
# ==============================

import joblib
import numpy as np
import os

# ------------------------------
# Safe Path Handling
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# ------------------------------
# Load Model Safely
# ------------------------------
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"Error loading model/scaler: {e}")

# Load once
model, scaler = load_artifacts()

# ------------------------------
# Prediction Function
# ------------------------------
def predict_heart_attack(input_data):
    """
    input_data: list of 13 feature values
    """

    if len(input_data) != 13:
        raise ValueError("Expected 13 input features")

    input_array = np.array(input_data).reshape(1, -1)

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Probability (if available)
    try:
        prob = model.predict_proba(input_scaled)[0][1]
    except:
        prob = None

    result = "High Risk" if prediction == 1 else "Low Risk"

    return result, prob