import pickle

from flask import Flask, render_template, request

from lib.registry import load_model

IRIS_TYPES = {0: "setosa", 1: "versicolor", 2: "virginica"}

app = Flask(__name__)


@app.route("/", methods=["GET"])
def iris_index():
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def result():
    if request.method == "POST":

        sepal_length = request.form["inputSepalLength"]
        sepal_width = request.form["inputSepalWidth"]
        petal_length = request.form["inputPetalLength"]
        petal_width = request.form["inputPetalWidth"]

        data = [
            [
                float(sepal_length),
                float(sepal_width),
                float(petal_length),
                float(petal_width),
            ]
        ]
        model = load_model()
        pred = model.predict(data)[0]

        return render_template("prediction.html", iris_type=IRIS_TYPES[pred])


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5000, debug=True)
