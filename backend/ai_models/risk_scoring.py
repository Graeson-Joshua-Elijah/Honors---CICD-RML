import pickle
import numpy as np
import sys

# Load trained reptile model
with open("ai_models/reptile_model.pkl", "rb") as f:
    model = pickle.load(f)

def calculate_risk(features):
    features = np.array(features).reshape(1, -1)
    prob = model.predict_proba(features)[0][1]
    return prob

if __name__ == "__main__":
    # Expecting: python risk_scoring.py build_time failed_tests
    if len(sys.argv) < 3:
        print("Usage: python risk_scoring.py <build_time> <failed_tests>")
        sys.exit(1)

    build_time = float(sys.argv[1])
    failed_tests = int(sys.argv[2])
    features = [build_time, failed_tests]

    risk_score = calculate_risk(features)
    print("Risk Score:", risk_score)

    # Fail pipeline if risk is high
    if risk_score > 0.7:
        print("⚠ Deployment blocked: High failure risk")
        sys.exit(1)
    else:
        print("✅ Deployment allowed")