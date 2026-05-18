"""Run a sample prediction and print human-readable output."""
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "model" / "diabetes_model.pkl"

# Example patient (similar to a high-risk row in the dataset)
SAMPLE = {
    "Gender": 0,
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}


def main() -> None:
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([SAMPLE])
    prediction = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    risk = float(proba[1]) * 100

    label = "Diabetes likely" if prediction == 1 else "No diabetes"
    print("=== Diabetes Prediction Output ===")
    print()
    print("Input features:")
    for name, value in SAMPLE.items():
        print(f"  {name}: {value}")
    print()
    print(f"Prediction: {prediction} ({label})")
    print(f"Risk score: {risk:.1f}%")
    print(f"Confidence (no diabetes): {proba[0] * 100:.1f}%")
    print(f"Confidence (diabetes): {proba[1] * 100:.1f}%")


if __name__ == "__main__":
    main()
