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

# --- 1. SETUP AUTH ---
# Cek apakah script ini jalan di dalam MLflow Run (GitHub Actions)?
active_run = mlflow.active_run()

if active_run is None:
    # HANYA Setup manual jika jalan lokal (Laptop)
    print("⚠️ Running Locally: Setup auth manual...")
    TOKEN = os.environ.get("DAGSHUB_TOKEN", "TOKEN_ASLI_ANDA_DISINI") 
    os.environ["MLFLOW_TRACKING_USERNAME"] = "GalihRasyid"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN
    try:
        add_app_token(TOKEN)
    except:
        pass
    dagshub.init(repo_owner='GalihRasyid', repo_name='submission_diabetes_GalihRasyid', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/GalihRasyid/submission_diabetes_GalihRasyid.mlflow")
    mlflow.set_experiment("Diabetes_Fixed_Model") # Set eksperimen hanya kalau lokal

# --- 2. LOAD DATA ---
print("📂 Memulai proses loading data...")
csv_filename = 'diabetes_clean.csv'

if os.path.exists(csv_filename):
    df = pd.read_csv(csv_filename)
else:
    raise FileNotFoundError(f"❌ File {csv_filename} tidak ditemukan!")

if 'Outcome' in df.columns:
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
elif 'target' in df.columns:
    X = df.drop('target', axis=1)
    y = df['target']
else:
    raise ValueError("Kolom Target tidak ditemukan!")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. TRAINING & LOGGING ---
print("🚀 Memulai Training Model...")

# FUNGSI UTAMA: Tentukan konteks run
# Jika sudah ada active_run (dari GitHub Actions), pakai itu. Jangan start_run() lagi!
if active_run:
    print(f"ℹ️ Menggunakan Active Run ID: {active_run.info.run_id}")
    # Tidak perlu 'with mlflow.start_run():' karena sudah otomatis aktif
    
    mlflow.sklearn.autolog(disable=True) # Matikan autolog (Revisi Reviewer)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"📊 Accuracy: {acc}")
    
    # Log Manual
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 100)
    
    # SIMPAN MODEL (Ini yang membuat folder 'model' muncul)
    print("💾 Menyimpan model ke MLflow Artifacts...")
    mlflow.sklearn.log_model(model, "model") 
    
    # Backup file lokal
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")

else:
    # Jika Lokal, kita harus start_run manual
    print("ℹ️ Membuat New Run (Lokal)...")
    with mlflow.start_run(run_name="Run_Lokal_Manual"):
        mlflow.sklearn.autolog(disable=True)
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("n_estimators", 100)
        
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "model.pkl")
        mlflow.log_artifact("model.pkl")

print("✅ Selesai. Cek DagsHub sekarang.")