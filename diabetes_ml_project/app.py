from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("diabetes_ml_project/model/diabetes_model.pkl")

HTML = """
<h1>GlucoNova Diabetes Risk Intelligence</h1>

<form method="POST">
Pregnancies: <input name="preg"><br><br>
Glucose: <input name="glucose"><br><br>
Blood Pressure: <input name="bp"><br><br>
Skin Thickness: <input name="skin"><br><br>
Insulin: <input name="insulin"><br><br>
BMI: <input name="bmi"><br><br>
Diabetes Pedigree Function: <input name="dpf"><br><br>
Age: <input name="age"><br><br>

<input type="submit" value="Predict">
</form>

<h2>{{result}}</h2>
"""

@app.route("/", methods=["GET","POST"])
def home():

    result=""

    if request.method=="POST":
        data=np.array([[
            float(request.form["preg"]),
            float(request.form["glucose"]),
            float(request.form["bp"]),
            float(request.form["skin"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            float(request.form["age"])
        ]])

        prediction=model.predict(data)

        if prediction[0]==1:
            result="Diabetes likely"
        else:
            result="No diabetes detected"

    return render_template_string(HTML,result=result)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)