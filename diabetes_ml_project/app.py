from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("diabetes_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    prediction = model.predict([[1,148,72,35,0,33.6,0.627,50]])

    if prediction[0]==1:
        result="Diabetes Likely"
    else:
        result="No Diabetes"

    return render_template(
        "index.html",
        prediction_text=result
    )

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)
