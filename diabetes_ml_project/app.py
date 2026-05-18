from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("diabetes_ml_project/model/diabetes_model.pkl")

@app.route("/", methods=["GET","POST"])
def home():

    result=""

    if request.method=="POST":

        data=np.array([[
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"])
        ]])

        prediction=model.predict(data)

        if prediction[0]==1:
            result="⚠ Diabetes likely"
        else:
            result="✅ No diabetes detected"

    return render_template(
        "index.html",
        result=result
    )


if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)