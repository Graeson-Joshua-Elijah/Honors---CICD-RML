# import numpy as np
# import pickle
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # --- Reptile meta-learning loop ---
# def reptile_train(tasks, steps=5, meta_iterations=50):
#     # Initialize a base model
#     X_dummy, y_dummy = tasks[0][0], tasks[0][1]
#     meta_model = LogisticRegression(max_iter=200)
#     meta_model.fit(X_dummy, y_dummy)  # initial fit ensures .classes_ is created

#     for it in range(meta_iterations):
#         task_models = []
#         for X_train, y_train, X_val, y_val in tasks:
#             # Copy a fresh LogisticRegression and fit on task data
#             task_model = LogisticRegression(max_iter=200)
#             task_model.fit(X_train, y_train)
#             task_models.append(task_model)

#         # Aggregate updates (meta step) by averaging coefficients
#         coefs = [m.coef_ for m in task_models]
#         intercepts = [m.intercept_ for m in task_models]

#         meta_model.coef_ = np.mean(coefs, axis=0)
#         meta_model.intercept_ = np.mean(intercepts, axis=0)
#         meta_model.classes_ = np.unique(np.concatenate([m.classes_ for m in task_models]))

#         if it % 10 == 0:
#             print(f"Meta-iteration {it}/{meta_iterations} complete")

#     return meta_model


# def generate_dummy_tasks(num_tasks=5, samples=100):
#     tasks = []
#     for _ in range(num_tasks):
#         X = np.random.randint(1, 100, size=(samples, 2))
#         y = np.random.randint(0, 2, size=(samples,))
#         split = int(0.8 * samples)
#         X_train, y_train = X[:split], y[:split]
#         X_val, y_val = X[split:], y[split:]
#         tasks.append((X_train, y_train, X_val, y_val))
#     return tasks


# if __name__ == "__main__":
#     # Generate synthetic tasks
#     tasks = generate_dummy_tasks(num_tasks=5)

#     # Train Reptile meta-model
#     meta_model = reptile_train(tasks, steps=3, meta_iterations=30)

#     # Evaluate on unseen task
#     X_test = np.random.randint(1, 100, size=(50, 2))
#     y_test = np.random.randint(0, 2, size=(50,))
#     y_pred = meta_model.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)
#     print("Reptile Meta-Learned Model Accuracy:", acc)

#     # Save model
#     with open("ai_models/reptile_model.pkl", "wb") as f:
#         pickle.dump(meta_model, f)

import kagglehub
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Step 1: Download and load dataset ---
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("rahuljangir78/ai-driven-cicd-pipeline-logs-dataset")
print("Path to dataset files:", path)

# Automatically find CSV file in downloaded folder
csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("No CSV files found in the dataset directory.")
data_path = csv_files[0]

print(f"Loading dataset: {data_path}")
df = pd.read_csv(data_path)

# --- Step 2: Preprocess dataset ---
# Drop missing values
df = df.dropna()

# Encode categorical columns if any
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Standardize numerical columns
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df)

# Assume the last column is the target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# --- Step 3: Create multiple tasks for Reptile ---
def create_tasks(X, y, num_tasks=5):
    tasks = []
    for _ in range(num_tasks):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
        tasks.append((X_train, y_train, X_val, y_val))
    return tasks

tasks = create_tasks(X, y, num_tasks=5)

# --- Step 4: Reptile meta-learning loop ---
def reptile_train(tasks, steps=5, meta_iterations=30):
    X_dummy, y_dummy = tasks[0][0], tasks[0][1]
    meta_model = LogisticRegression(max_iter=300)
    meta_model.fit(X_dummy, y_dummy)

    for it in range(meta_iterations):
        task_models = []
        for X_train, y_train, X_val, y_val in tasks:
            task_model = LogisticRegression(max_iter=300)
            task_model.fit(X_train, y_train)
            task_models.append(task_model)

        # Meta-aggregation: average task weights
        coefs = [m.coef_ for m in task_models]
        intercepts = [m.intercept_ for m in task_models]

        meta_model.coef_ = np.mean(coefs, axis=0)
        meta_model.intercept_ = np.mean(intercepts, axis=0)
        meta_model.classes_ = np.unique(np.concatenate([m.classes_ for m in task_models]))

        if it % 5 == 0:
            print(f"Meta-iteration {it}/{meta_iterations} complete")

    return meta_model

# --- Step 5: Train and evaluate ---
print("\nTraining Reptile meta-learning model...")
meta_model = reptile_train(tasks, steps=5, meta_iterations=30)

# Evaluate on a held-out test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
y_pred = meta_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nReptile Meta-Learned Model Accuracy:", round(acc, 4))

# --- Step 6: Save model ---
os.makedirs("ai_models", exist_ok=True)
with open("ai_models/reptile_model.pkl", "wb") as f:
    pickle.dump(meta_model, f)
print("Model saved at: ai_models/reptile_model.pkl")
