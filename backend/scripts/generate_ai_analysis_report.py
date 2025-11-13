import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import matplotlib.pyplot as plt
import pandas as pd
from ai_models.failure_prediction import predict_failure
from ai_models.resource_analysis import analyze_trends
from ai_models.risk_scoring import calculate_risk

# Create output folder
os.makedirs("output/ai_analysis_report", exist_ok=True)

# --- Simulate Inputs for AI Models ---
dummy_build_data = {"build_size": 120, "commit_count": 15, "test_pass_rate": 0.92}
dummy_risk_data = {"commit_size": 220, "author_experience": 3, "test_coverage": 85}

# --- Run AI Modules ---
failure_prob = predict_failure(dummy_build_data)
resource_summary = analyze_trends()
risk_score = calculate_risk(dummy_risk_data)

# --- Combine all results ---
report = {
    "AI_Features": {
        "Failure_Prediction": failure_prob,
        "Resource_Utilization_Analysis": resource_summary,
        "Deployment_Risk_Scoring": risk_score,
        "Auto_Rollback": "Triggered if risk_score > 0.7 or failure_prob > 0.8",
        "Root_Cause_Analysis": "Logs are summarized using NLP (future integration)"
    }
}

# --- Save JSON Report ---
json_path = "output/ai_analysis_report/ai_report.json"
with open(json_path, "w") as f:
    json.dump(report, f, indent=4)

# --- Create Graphs ---
# 1. Resource Utilization
if resource_summary and "avg_cpu" in resource_summary:
    metrics_df = pd.DataFrame([resource_summary])
    metrics_df.plot(kind="bar", figsize=(8,5), title="Resource Utilization Analysis")
    plt.ylabel("Percentage")
    plt.tight_layout()
    plt.savefig("output/ai_analysis_report/resource_utilization.png")

# 2. Deployment Risk
plt.figure(figsize=(5,4))
plt.bar(["Risk Score"], [risk_score], color='red' if risk_score > 0.7 else 'green')
plt.title("Deployment Risk Scoring")
plt.ylim(0, 1)
plt.ylabel("Risk Level (0-1)")
plt.tight_layout()
plt.savefig("output/ai_analysis_report/risk_score.png")

print("✅ AI Analysis Report generated successfully.")
print(f"📁 Saved in: {os.path.abspath('output/ai_analysis_report')}")