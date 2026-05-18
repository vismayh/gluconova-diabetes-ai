"""Train RandomForest on Pima Indians Diabetes dataset and save model."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
)
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"
README_PATH = Path(__file__).resolve().parent / "README.txt"


def main() -> None:
    df = pd.read_csv(DATA_URL)
    # Pima dataset is female patients; Gender=0 (female), used as model feature at inference
    df["Gender"] = 0
    feature_cols = [
        "Gender",
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    X = df[feature_cols]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200, random_state=42, max_features="sqrt"
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    README_PATH.write_text(
        f"Diabetes model trained successfully\n"
        f"Accuracy:{accuracy:.4f}\n"
        f"Dataset rows:{len(df)}\n",
        encoding="utf-8",
    )

    print("Diabetes model trained successfully")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Dataset rows: {len(df)}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
