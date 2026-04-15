from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from predict import predict_churn, model
import traceback

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# 🔥 NEW: expose metrics from training
MODEL_METRICS = {
    "roc_auc": 0.9963,
    "accuracy": 0.9680,
    "precision": 0.9713,
    "recall": 0.9313,
    "f1": 0.9509,
    "model": "xgboost"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify(MODEL_METRICS)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print(f"Données reçues: {data}")

        required_keys = ["Recency", "Frequency", "MonetaryTotal", "AvgBasketValue", "CustomerTenureDays"]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Clé manquante: {key}"}), 400

        result = predict_churn(data)

        return jsonify({"probability": result})

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "prediction failed"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)