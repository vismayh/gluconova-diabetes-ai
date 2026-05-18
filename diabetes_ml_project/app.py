from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>GlucoNova Diabetes Risk Intelligence</h1>
    <p>AI Diabetes Prediction System is running successfully</p>
    <a href='/predict'>Go to Prediction</a>
    """

@app.route("/predict")
def predict():
    return "Prediction page coming from your ML model"

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)