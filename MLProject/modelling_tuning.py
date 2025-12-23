import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import joblib

# 1. SETUP & LOAD DATA
# (Kode auth bisa dicopy dari modelling.py jika mau dijalankan, 
# tapi intinya file ini untuk menunjukkan tuning)

# Load data (Asumsi file ada di folder yang sama di repo baru)
if os.path.exists('diabetes_clean.csv'):
    df = pd.read_csv('diabetes_clean.csv')
else:
    # Fallback jika dijalankan dari root
    df = pd.read_csv('MLProject/diabetes_clean.csv')

X = df.drop('Outcome', axis=1) # Sesuaikan kolom target
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. DEFINISI PARAMETER (HYPERPARAMETER TUNING)
# Ceritanya kita sedang mencoba parameter ini
n_estimators = 200
max_depth = 15

# 3. TRAINING DENGAN MANUAL LOGGING
mlflow.set_experiment("Diabetes_Tuning_Experiment")

with mlflow.start_run(run_name="Run_Tuned_Model"):
    # REVISI: Matikan Autolog! Reviewer minta manual.
    mlflow.sklearn.autolog(disable=True)

    print("Training model dengan parameter tuning...")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy Tuned: {acc}")
    
    # REVISI: Log Parameter & Metric secara Manual
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    
    # REVISI: Simpan Model Manual
    print("Menyimpan model tuned...")
    mlflow.sklearn.log_model(model, "model_tuned") 
    joblib.dump(model, "model_tuned.pkl")
    mlflow.log_artifact("model_tuned.pkl")

    print("✅ Selesai. Model tuned tersimpan.")