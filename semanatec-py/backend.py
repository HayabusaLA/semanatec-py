from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import joblib
import os
from os import path
import csv
from csv import DictReader

dt = joblib.load("./static/dt.joblib")

app = Flask(__name__)
CORS(app)

@app.route("/Hola", methods=["GET"])
def inicio():
    return "Hola mundo"

@app.route("/predict_json", methods=["POST"])
def predict_json():
    data = request.json
    x = [[float(data["ph"]), float(data["sulphates"]), float(data["alcohol"])]]

    y_pred = dt.predict(x)
    print(y_pred)
    return jsonify({"result": y_pred[0]})

@app.route("/predict_form", methods=["POST"])
def predict_form():
    data = request.form
    x = [[float(data["ph"]), float(data["sulphates"]), float(data["alcohol"])]]

    y_pred = dt.predict(x)
    print(y_pred)
    return jsonify({"result": y_pred[0]})

@app.route("/predict_file", methods=["POST"])
def predict_file():
    file = request.files["archivo"]
    filename = secure_filename(file.filename)
    path = os.path.join(os.getcwd(), "static", filename)
    file.save(path)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        X = [[float(row["ph"]), float(row["sulphates"]), float(row["alcohol"])] for row in reader]
        y_pred = dt.predict(X)
        return jsonify({"results": [{"key": f"{k}", "result": f"{v}"} for k, v in enumerate(y_pred)]})


if __name__  == "__main__":
    app.run(host="0.0.0.0", debug = False, port = 8081)