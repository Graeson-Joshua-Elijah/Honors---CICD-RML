import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Reptile meta-learning loop ---
def reptile_train(tasks, steps=5, meta_iterations=50):
    # Initialize a base model
    X_dummy, y_dummy = tasks[0][0], tasks[0][1]
    meta_model = LogisticRegression(max_iter=200)
    meta_model.fit(X_dummy, y_dummy)  # initial fit ensures .classes_ is created

    for it in range(meta_iterations):
        task_models = []
        for X_train, y_train, X_val, y_val in tasks:
            # Copy a fresh LogisticRegression and fit on task data
            task_model = LogisticRegression(max_iter=200)
            task_model.fit(X_train, y_train)
            task_models.append(task_model)

        # Aggregate updates (meta step) by averaging coefficients
        coefs = [m.coef_ for m in task_models]
        intercepts = [m.intercept_ for m in task_models]

        meta_model.coef_ = np.mean(coefs, axis=0)
        meta_model.intercept_ = np.mean(intercepts, axis=0)
        meta_model.classes_ = np.unique(np.concatenate([m.classes_ for m in task_models]))

        if it % 10 == 0:
            print(f"Meta-iteration {it}/{meta_iterations} complete")

    return meta_model


def generate_dummy_tasks(num_tasks=5, samples=100):
    tasks = []
    for _ in range(num_tasks):
        X = np.random.randint(1, 100, size=(samples, 2))
        y = np.random.randint(0, 2, size=(samples,))
        split = int(0.8 * samples)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]
        tasks.append((X_train, y_train, X_val, y_val))
    return tasks


if __name__ == "__main__":
    # Generate synthetic tasks
    tasks = generate_dummy_tasks(num_tasks=5)

    # Train Reptile meta-model
    meta_model = reptile_train(tasks, steps=3, meta_iterations=30)

    # Evaluate on unseen task
    X_test = np.random.randint(1, 100, size=(50, 2))
    y_test = np.random.randint(0, 2, size=(50,))
    y_pred = meta_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Reptile Meta-Learned Model Accuracy:", acc)

    # Save model
    with open("ai_models/reptile_model.pkl", "wb") as f:
        pickle.dump(meta_model, f)

