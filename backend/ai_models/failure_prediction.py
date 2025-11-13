import pickle
import numpy as np

# Load trained reptile model
with open("ai_models/reptile_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_failure(features):
    """
    Predicts failure risk using the Reptile meta-learned model.
    :param features: list/array of feature values (length must match training features)
    :return: int (0 = No Failure, 1 = Failure)
    """
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]

if __name__ == "__main__":
    test_features = [50, 20]  # Example features
    prediction = predict_failure(test_features)
    print("Failure Prediction:", prediction)
