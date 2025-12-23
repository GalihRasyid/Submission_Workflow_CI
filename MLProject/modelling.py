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
    TOKEN = os.environ.get("DAGSHUB_TOKEN", "d1f669853cea910190197feb84d64f7cb5691026") 
    os.environ["MLFLOW_TRACKING_USERNAME"] = "GalihRasyid"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN
    try:
        add_app_token(TOKEN)
    except:
        pass
    dagshub.init(repo_owner='GalihRasyid', repo_name='submission_diabetes_GalihRasyid', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/GalihRasyid/submission_diabetes_GalihRasyid.mlflow")
    # Set eksperimen hanya jika lokal, biar ga konflik sama CI
    mlflow.set_experiment("Diabetes_Fixed_Model") 

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

# FUNGSI TRAINING (Dibungkus agar bisa dipanggil di kondisi apapun)
def train_and_log():
    # Matikan autolog (Sesuai Revisi)
    mlflow.sklearn.autolog(disable=True)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"📊 Accuracy: {acc}")
    
    # Log Manual
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 100)
    
    # --- BAGIAN KRUSIAL (PEMBUAT FOLDER MODEL) ---
    print("💾 Menyimpan model ke MLflow Artifacts...")
    
    # 1. INI YANG MEMBUAT FOLDER 'model' LENGKAP (MLmodel, conda.yaml, dll)
    mlflow.sklearn.log_model(model, artifact_path="model") 
    
    # 2. INI YANG MEMBUAT FILE 'model.pkl' BIASA (BACKUP)
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")

    print("✅ Model berhasil disimpan (Folder 'model' & File 'model.pkl').")

# EKSEKUSI
if active_run:
    print(f"ℹ️ Menggunakan Active Run ID: {active_run.info.run_id}")
    # Langsung jalan tanpa start_run baru
    train_and_log()
else:
    print("ℹ️ Membuat New Run (Lokal)...")
    with mlflow.start_run(run_name="Run_Lokal_Manual"):
        train_and_log()