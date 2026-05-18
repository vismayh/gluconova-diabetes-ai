@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    values = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['bloodpressure']),
        float(request.form['skinthickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age'])
    ]

    prediction = model.predict([values])

    result = "Diabetes Likely" if prediction[0]==1 else "No Diabetes Detected"

    return render_template(
        "index.html",
        prediction_text=result
    )