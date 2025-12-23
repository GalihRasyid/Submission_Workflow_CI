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
# Cek apakah sedang berjalan di dalam MLflow Run (GitHub Actions)
is_mlflow_run = mlflow.active_run() is not None

if not is_mlflow_run:
    # Setup manual HANYA jika dijalankan lokal di laptop (bukan oleh mlflow run)
    print("⚠️ Running Locally: Setup auth manual...")
    TOKEN = os.environ.get("DAGSHUB_TOKEN", "d1f669853cea910190197feb84d64f7cb5691026")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "GalihRasyid"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN
    try:
        add_app_token(TOKEN)
    except:
        pass
    dagshub.init(repo_owner='GalihRasyid', repo_name='submission_diabetes_GalihRasyid', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/GalihRasyid/submission_diabetes_GalihRasyid.mlflow")
    
    # Hanya set experiment jika lokal
    mlflow.set_experiment("Diabetes_Fixed_Model")

# --- 2. LOAD DATA ---
print("📂 Memulai proses loading data...")
csv_filename = 'diabetes_clean.csv'

if os.path.exists(csv_filename):
    df = pd.read_csv(csv_filename)
    print(f"✅ Dataset ditemukan: {csv_filename}")
else:
    raise FileNotFoundError(f"❌ Gagal load {csv_filename}. Pastikan file ada di folder yang sama.")

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

# LOGIKA PENTING UNTUK MENGHINDARI ERROR KONFLIK ID
# Jika is_mlflow_run = True, berarti kita sudah punya Run ID dari 'mlflow run' (GitHub Actions)
# Jadi kita pakai mlflow.start_run() tanpa argumen agar dia masuk ke Run ID tersebut.

if is_mlflow_run:
    print("ℹ️ Terdeteksi Active Run dari MLflow Project. Menggunakan Run ID yang sudah ada.")
    run_context = mlflow.start_run()
else:
    print("ℹ️ Tidak ada Active Run. Membuat Run baru.")
    run_context = mlflow.start_run(run_name="Run_Manual_Log")

with run_context:
    # Matikan autolog (Sesuai Revisi)
    mlflow.sklearn.autolog(disable=True)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"📊 Accuracy: {acc}")
    
    # Logging Manual (Sesuai Revisi)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Simpan Model Manual (Sesuai Revisi)
    print("💾 Menyimpan model ke MLflow Artifacts...")
    mlflow.sklearn.log_model(model, "model") 
    
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")

    print("✅ Model berhasil disimpan.")