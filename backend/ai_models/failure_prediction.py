import pickle
import numpy as np
import os

# Resolve absolute path to ai_models folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "reptile_model.pkl")

# Try loading model safely
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"[WARNING] Model file not found: {MODEL_PATH}. Using fallback logic.")


def extract_features(build_data: dict):
    """
    Converts build data dictionary into a feature vector.
    Required keys: build_size, commit_count, test_pass_rate
    """

    return [
        build_data.get("build_size", 0),
        build_data.get("commit_count", 0),
        build_data.get("test_pass_rate", 0)
    ]


def predict_failure(build_data):
    """
    Predicts failure probability using the loaded model OR returns fallback value.
    :param build_data: dict with build metrics
    :return: float (0–1 probability score)
    """
    features = np.array(extract_features(build_data)).reshape(1, -1)

    if model:
        return float(model.predict(features)[0])

    # Fallback if model is missing — avoid CI crash
    return 0.5   # neutral probability


if __name__ == "__main__":
    dummy = {"build_size": 50, "commit_count": 20, "test_pass_rate": 0.9}
    prediction = predict_failure(dummy)
    print("Failure Prediction:", prediction)
