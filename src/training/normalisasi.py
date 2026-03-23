# ==========================================
# 🔧 BUAT ULANG SCALER
# ==========================================
# Jalankan ini jika scaler.pkl hilang
#   python src/make_scaler.py
# ==========================================

import numpy as np
from pathlib import Path
import pickle
import gc
from tqdm import tqdm

TRAIN_DIR   = Path('data/processed/dataset_npz/TRAIN')
SCALER_PATH = Path('models/scaler.pkl')
N_CHANNELS  = 8
N_TIMES     = 128

Path('models').mkdir(exist_ok=True)

print("📐 Menghitung scaler dari TRAIN...")

n_features = N_CHANNELS * N_TIMES
mean  = np.zeros(n_features, dtype=np.float64)
M2    = np.zeros(n_features, dtype=np.float64)
count = 0

files = sorted(TRAIN_DIR.glob("*.npz"))
print(f"   Total file: {len(files)}")

for f in tqdm(files, desc="Fit scaler"):
    d      = np.load(f)
    X      = d['X'].astype(np.float64)
    X_flat = X.reshape(len(X), -1)

    for x in X_flat:
        count  += 1
        delta   = x - mean
        mean   += delta / count
        delta2  = x - mean
        M2     += delta * delta2

    del X, X_flat, d
    gc.collect()

std = np.sqrt(M2 / count)
std[std < 1e-10] = 1.0

scaler = {
    'mean': mean.astype(np.float32),
    'std':  std.astype(np.float32)
}

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Scaler disimpan → {SCALER_PATH}")
print(f"   Sekarang jalankan: python src/train_eegnet.py")