import os
import joblib
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dagshub.auth import add_app_token

# 1. SETUP AUTH (Cek Env Var agar aman di CI/CD)
if "MLFLOW_TRACKING_URI" not in os.environ:
    # Setup manual lokal (jika dijalankan di laptop)
    TOKEN = os.environ.get("DAGSHUB_TOKEN", "TOKEN_DAGSHUB_ANDA_DISINI") 
    os.environ["MLFLOW_TRACKING_USERNAME"] = "GalihRasyid"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN
    try:
        add_app_token(TOKEN)
    except:
        pass
    dagshub.init(repo_owner='GalihRasyid', repo_name='submission_diabetes_GalihRasyid', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/GalihRasyid/submission_diabetes_GalihRasyid.mlflow")

# 2. LOAD DATA
# Pastikan file csv ada di folder yang sama di repo Workflow nanti
df = pd.read_csv('diabetes_clean.csv') 

X = df.drop('Outcome', axis=1) # Sesuaikan nama kolom target
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAINING
mlflow.set_experiment("Diabetes_Fixed_Model")

with mlflow.start_run(run_name="Run_Manual_Log"):
    # MATIKAN AUTOLOG (Agar kita bisa kontrol penuh penyimpanannya)
    mlflow.sklearn.autolog(disable=True)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")
    
    # MANUAL LOGGING METRICS
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 100)
    
    # --- BAGIAN PENTING (SOLUSI REVISI) ---
    print("Simpan model ke DagsHub...")
    
    # 1. Ini yang membuat folder 'model' muncul di Artifacts DagsHub
    mlflow.sklearn.log_model(model, "model")
    
    # 2. Ini backup file lokal
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl") 
    
    print("Selesai.")