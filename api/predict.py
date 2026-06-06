from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")

model = joblib.load(MODEL_PATH)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        input_data = np.array([[
            float(data["study_hours"]),
            float(data["focus_score"]),
            float(data["assignments"]),
            float(data["attendance"]),
            float(data["phone_hours"]),
            float(data["stress_level"]),
            float(data["sleep_hours"])
        ]])

        prediction = model.predict(input_data)[0]
        prediction = min(max(prediction, 15.0), 99.9)

        return jsonify({
            "success": True,
            "predicted_score": round(float(prediction), 2),
            "score": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500