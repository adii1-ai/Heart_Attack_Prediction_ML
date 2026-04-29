# ==============================
# Heart Attack Model Training (Fixed)
# ==============================

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# Safe Path Handling
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "HeartData.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "features.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------
# 1. Load Data
# ------------------------------
def load_data():
    return pd.read_csv(DATA_PATH)

# ------------------------------
# 2. Preprocess Data
# ------------------------------
def preprocess_data(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

# ------------------------------
# 3. Train Models
# ------------------------------
def train_models(X_train, y_train):
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=42)
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models

# ------------------------------
# 4. Evaluate Models
# ------------------------------
def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n{name.upper()} RESULTS")
        print("Accuracy:", acc)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        results[name] = acc

    return results

# ------------------------------
# 5. Hyperparameter Tuning
# ------------------------------
def tune_logistic(X_train, y_train):
    param_grid = {"C": [0.1, 1, 10]}

    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    grid.fit(X_train, y_train)

    print("Best Logistic Params:", grid.best_params_)
    return grid.best_estimator_

# ------------------------------
# 6. Save Artifacts
# ------------------------------
def save_artifacts(model, scaler, feature_names):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names, FEATURE_PATH)

    print(f"Model saved at: {MODEL_PATH}")
    print(f"Scaler saved at: {SCALER_PATH}")
    print(f"Features saved at: {FEATURE_PATH}")

# ------------------------------
# 7. Main Pipeline
# ------------------------------
def main():
    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    X, y = preprocess_data(df)

    # Save feature order
    feature_names = X.columns.tolist()

    # Train-test split (with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling (correct way)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training models...")
    models = train_models(X_train, y_train)

    print("Evaluating models...")
    results = evaluate_models(models, X_test, y_test)

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    print(f"\nBest Model: {best_model_name}")

    if best_model_name == "logistic":
        best_model = tune_logistic(X_train, y_train)

    save_artifacts(best_model, scaler, feature_names)

    print("Training complete!")

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    main()