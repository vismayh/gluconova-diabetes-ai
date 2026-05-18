from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "diabetes_ml_project"
    / "model"
    / "diabetes_model.pkl"
)

app = FastAPI(title="Diabetes Prediction API")
model = joblib.load(MODEL_PATH)


class PatientInput(BaseModel):
    pregnancies: int = Field(..., ge=0, alias="Pregnancies")
    glucose: float = Field(..., ge=0, alias="Glucose")
    blood_pressure: float = Field(..., ge=0, alias="BloodPressure")
    skin_thickness: float = Field(..., ge=0, alias="SkinThickness")
    insulin: float = Field(..., ge=0, alias="Insulin")
    bmi: float = Field(..., gt=0, alias="BMI")
    diabetes_pedigree_function: float = Field(
        ..., ge=0, alias="DiabetesPedigreeFunction"
    )
    age: int = Field(..., ge=0, alias="Age")

    model_config = {"populate_by_name": True}


@app.get("/")
def root() -> dict:
    return {
        "message": "Diabetes prediction API",
        "predict_endpoint": "POST /predict",
    }


@app.post("/predict")
def predict(patient: PatientInput) -> dict:
    row = [
        patient.pregnancies,
        patient.glucose,
        patient.blood_pressure,
        patient.skin_thickness,
        patient.insulin,
        patient.bmi,
        patient.diabetes_pedigree_function,
        patient.age,
    ]
    outcome = int(model.predict([row])[0])
    proba = model.predict_proba([row])[0]
    return {
        "prediction": outcome,
        "label": "Diabetes likely" if outcome == 1 else "No diabetes",
        "risk_percent": round(float(proba[1]) * 100, 2),
        "probabilities": {
            "no_diabetes": round(float(proba[0]) * 100, 2),
            "diabetes": round(float(proba[1]) * 100, 2),
        },
        "input": patient.model_dump(by_alias=True),
    }
