import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "diabetes_ml_project" / "model" / "diabetes_model.pkl"
README_PATH = BASE_DIR.parent / "diabetes_ml_project" / "README.txt"
STATIC_DIR = BASE_DIR / "static"
COUNTER_PATH = BASE_DIR / "data" / "prediction_count.json"

app = FastAPI(title="GlucoNova — Diabetes Risk Intelligence")
_model_cache: object | None = None
_model_mtime: float | None = None

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_model():
    global _model_cache, _model_mtime
    mtime = MODEL_PATH.stat().st_mtime
    if _model_cache is None or _model_mtime != mtime:
        _model_cache = joblib.load(MODEL_PATH)
        _model_mtime = mtime
    return _model_cache


def _gender_to_code(gender: str) -> int:
    g = gender.strip().lower()
    if g == "male":
        return 1
    return 0  # female and other


class PatientInput(BaseModel):
    gender: Literal["female", "male", "other"] = Field(..., alias="Gender")
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


def _get_prediction_count() -> int:
    if not COUNTER_PATH.exists():
        return 0
    try:
        data = json.loads(COUNTER_PATH.read_text(encoding="utf-8"))
        return int(data.get("count", 0))
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0


def _increment_prediction_count() -> int:
    COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    count = _get_prediction_count() + 1
    COUNTER_PATH.write_text(
        json.dumps({"count": count}, indent=2),
        encoding="utf-8",
    )
    return count


def _read_readme() -> dict[str, str]:
    text = README_PATH.read_text(encoding="utf-8")
    accuracy = "73.38%"
    rows = "768"
    for line in text.splitlines():
        if line.lower().startswith("accuracy:"):
            val = line.split(":", 1)[1].strip()
            accuracy = f"{float(val) * 100:.2f}%"
        if line.lower().startswith("dataset rows:"):
            rows = line.split(":", 1)[1].strip()
    return {"accuracy": accuracy, "dataset_rows": rows}


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/model-info")
def model_info() -> dict:
    meta = _read_readme()
    model = get_model()
    return {
        "brand": "GlucoNova",
        "tagline": "Diabetes risk intelligence",
        "accuracy": meta["accuracy"],
        "dataset_rows": meta["dataset_rows"],
        "prediction_count": _get_prediction_count(),
        "features": list(model.feature_names_in_),
    }


@app.post("/predict")
def predict(patient: PatientInput) -> dict:
    gender_code = _gender_to_code(patient.gender)
    pregnancies = 0 if patient.gender == "male" else patient.pregnancies
    model = get_model()
    row = {
        "Gender": gender_code,
        "Pregnancies": pregnancies,
        "Glucose": patient.glucose,
        "BloodPressure": patient.blood_pressure,
        "SkinThickness": patient.skin_thickness,
        "Insulin": patient.insulin,
        "BMI": patient.bmi,
        "DiabetesPedigreeFunction": patient.diabetes_pedigree_function,
        "Age": patient.age,
    }
    X = pd.DataFrame([row])
    outcome = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    prediction_count = _increment_prediction_count()
    return {
        "prediction": outcome,
        "prediction_count": prediction_count,
        "label": "Diabetes likely" if outcome == 1 else "No diabetes",
        "risk_percent": round(float(proba[1]) * 100, 2),
        "probabilities": {
            "no_diabetes": round(float(proba[0]) * 100, 2),
            "diabetes": round(float(proba[1]) * 100, 2),
        },
        "input": {
            **patient.model_dump(by_alias=True),
            "Pregnancies": pregnancies,
        },
        "gender": patient.gender,
    }
