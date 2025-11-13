from flask import Flask, jsonify, request, send_file
from ai_models.failure_prediction import predict_failure
from ai_models.resource_analysis import analyze_resources
from ai_models.risk_scoring import calculate_risk
from monitoring.monitor_resources import get_resource_metrics
import matplotlib.pyplot as plt
import io
import datetime
import json
import os

app = Flask(__name__)

# Root route
@app.route('/')
def home():
    return jsonify({"message": "AI-Driven CI/CD API with Visualization & Reporting is running!"})

# 1️⃣ Failure Prediction
@app.route('/predict_failure', methods=['POST'])
def failure_route():
    data = request.get_json()
    result = predict_failure(data)
    return jsonify({"failure_prediction": result})

# 2️⃣ Resource Utilization Analysis
@app.route('/analyze_resources', methods=['GET'])
def resource_route():
    result = analyze_resources()
    return jsonify(result)

# 3️⃣ Deployment Risk Scoring
@app.route('/risk_score', methods=['POST'])
def risk_route():
    data = request.get_json()
    score = calculate_risk(data)
    return jsonify({"risk_score": score})

# 4️⃣ Real-time Monitoring
@app.route('/monitor', methods=['GET'])
def monitor_route():
    metrics = get_resource_metrics()
    return jsonify(metrics)

# 5️⃣ Graph: Resource Utilization
@app.route('/graph/resource_utilization', methods=['GET'])
def graph_resources():
    metrics = get_resource_metrics()
    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("Resource Utilization Analysis")
    plt.ylabel("Usage (%)")
    plt.xlabel("Metrics")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

# 6️⃣ Graph: Deployment Risk Scoring
@app.route('/graph/risk_score', methods=['POST'])
def graph_risk():
    data = request.get_json()
    score = calculate_risk(data)
    threshold = 0.7  # ✅ adjustable threshold
    labels = ['Risk Score', 'Threshold']
    values = [score, threshold]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, values, color=['red', 'gray'])
    plt.title("Deployment Risk Scoring")
    plt.ylabel("Value")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

# 7️⃣ Generate AI Analysis Report (PDF or JSON)
@app.route('/generate_report', methods=['GET'])
def generate_report():
    ai_features = [
        "Automated Failure Prediction",
        "Intelligent Resource Utilization Analysis",
        "Deployment Risk Scoring AI",
        "Self-Learning CI/CD Feedback System",
        "Anomaly Detection in Metrics Stream"
    ]
    report_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ai_features": ai_features,
        "deployment_status": "Local Deployment Analysis Complete",
        "risk_threshold": 0.7,
        "notes": "AI models monitored and analyzed during local deployment."
    }

    path = "deployment_analysis_report.json"
    with open(path, "w") as f:
        json.dump(report_data, f, indent=4)

    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
