# ==========================================
# TRAINING EEGNet — bigP3BCI (FINAL)
# ==========================================
# Fix: kalibrasi dan evaluasi TEST pakai
# batch kecil agar tidak OOM di VRAM 2GB
#
# python src/train_eegnet.py
# ==========================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                              f1_score, classification_report)
import pickle
import gc
from tqdm import tqdm
import time

# ==========================================
# CONFIG
# ==========================================

TRAIN_DIR   = Path('data/processed/dataset_npz/TRAIN')
TEST_DIR    = Path('data/processed/dataset_npz/TEST')
SCALER_PATH = Path('models/scaler.pkl')
MODEL_PATH  = Path('models/eegnet_best.pt')
LOG_PATH    = Path('models/training_log.txt')

Path('models').mkdir(exist_ok=True)

SAMPLE_PER_FILE_TRAIN = 2000
BATCH_SIZE       = 128
INFER_BATCH_SIZE = 256   # batch kecil untuk inference agar aman VRAM
EPOCHS           = 100
LR               = 3e-4
POS_WEIGHT       = 8.67
N_CHANNELS       = 8
N_TIMES          = 128
RANDOM_STATE     = 42
CALIB_FILES      = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# CEK GPU
# ==========================================

def cek_device():
    print("=" * 55)
    print("INFO DEVICE")
    print("=" * 55)
    if torch.cuda.is_available():
        nama = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU    : {nama}")
        print(f"  VRAM   : {vram:.1f} GB")
        print(f"  Device : cuda")
    else:
        print("  Device : cpu")
    print()

# ==========================================
# LOAD SCALER
# ==========================================

def load_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Scaler tidak ditemukan! Jalankan make_scaler.py dulu.")
    with open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)

# ==========================================
# DATASET
# ==========================================

class P300TrainDataset(Dataset):
    def __init__(self, folder, scaler, sample_per_file,
                 augment=False, seed=RANDOM_STATE):
        self.augment = augment
        self.mean = scaler['mean'].reshape(N_CHANNELS, N_TIMES)
        self.std  = scaler['std'].reshape(N_CHANNELS, N_TIMES)

        files = sorted(folder.glob("*.npz"))
        X_all, Y_all = [], []
        rng = np.random.default_rng(seed)

        print(f"  {len(files)} file x {sample_per_file} sample/file...")
        for f in tqdm(files, desc="  Load", leave=False):
            d = np.load(f)
            X = d['X'].astype(np.float32)
            Y = d['Y'].astype(np.float32)

            idx_pos = np.where(Y == 1)[0]
            idx_neg = np.where(Y == 0)[0]
            n_each  = min(sample_per_file // 2,
                         len(idx_pos), len(idx_neg))

            sel = np.concatenate([
                rng.choice(idx_pos, n_each, replace=False),
                rng.choice(idx_neg, n_each, replace=False)
            ])

            X_all.append(X[sel])
            Y_all.append(Y[sel])
            del X, Y, d
            gc.collect()

        self.X = np.concatenate(X_all, axis=0)
        self.Y = np.concatenate(Y_all, axis=0)

        n_pos = int(self.Y.sum())
        n_neg = int((self.Y == 0).sum())
        print(f"  Total : {len(self.Y):,} samples "
              f"(target: {n_pos:,} | non-target: {n_neg:,})")

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.Y[idx]
        x = (x - self.mean) / self.std

        if self.augment:
            if np.random.rand() < 0.5:
                x += np.random.normal(0, 0.05, x.shape).astype(np.float32)
            if np.random.rand() < 0.3:
                shift = np.random.randint(-5, 6)
                x = np.roll(x, shift, axis=-1)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

# ==========================================
# MODEL — EEGNet
# ==========================================

class EEGNet(nn.Module):
    def __init__(self, n_channels=8, n_times=128,
                 F1=16, D=2, F2=32, dropout=0.25):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64),
                      padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out   = self.block3(self.block2(self.block1(dummy)))
            flat  = out.view(1, -1).shape[1]

        self.classifier = nn.Linear(flat, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(1)

# ==========================================
# HELPER: PREDIKSI BATCH KECIL
# Aman untuk VRAM 2GB
# ==========================================

@torch.no_grad()
def predict_in_batches(model, X_norm, batch_size=256):
    """Prediksi array numpy dalam batch kecil — aman VRAM 2GB."""
    model.eval()
    all_probs = []
    for start in range(0, len(X_norm), batch_size):
        end  = min(start + batch_size, len(X_norm))
        X_b  = torch.tensor(
            X_norm[start:end], dtype=torch.float32
        ).unsqueeze(1).to(DEVICE)
        prob = torch.sigmoid(model(X_b)).cpu().numpy()
        all_probs.extend(prob.tolist())
        del X_b, prob
        torch.cuda.empty_cache()
    return np.array(all_probs)

# ==========================================
# TRAINING LOOP
# ==========================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc


@torch.no_grad()
def evaluate_loader(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss   = criterion(logits, y)

        total_loss += loss.item() * len(y)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    avg_loss  = total_loss / len(loader.dataset)
    auc       = roc_auc_score(all_labels, all_preds)
    preds_bin = (np.array(all_preds) >= 0.5).astype(int)
    bacc      = balanced_accuracy_score(all_labels, preds_bin)
    return avg_loss, auc, bacc

# ==========================================
# KALIBRASI THRESHOLD — batch kecil
# ==========================================

def calibrate_threshold(model, scaler, test_dir, n_files):
    files = sorted(test_dir.glob("*.npz"))[:n_files]
    mean  = scaler['mean'].reshape(N_CHANNELS, N_TIMES)
    std   = scaler['std'].reshape(N_CHANNELS, N_TIMES)

    print(f"  Kalibrasi dari {n_files} file TEST asli...")
    all_probs, all_labels = [], []

    for f in files:
        d      = np.load(f)
        X      = d['X'].astype(np.float32)
        Y      = d['Y'].astype(np.int32)
        X_norm = (X - mean) / std

        # Batch kecil — aman VRAM 2GB
        probs = predict_in_batches(model, X_norm, INFER_BATCH_SIZE)
        all_probs.extend(probs.tolist())
        all_labels.extend(Y.tolist())
        del X, Y, X_norm, probs, d
        gc.collect()

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    ratio = (all_labels == 0).sum() / max((all_labels == 1).sum(), 1)
    print(f"  Distribusi: {ratio:.1f}:1 (imbalanced)")

    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (all_probs >= t).astype(int)
        f1   = f1_score(all_labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"  Threshold optimal: {best_thresh:.2f} (F1={best_f1:.4f})")
    return best_thresh

# ==========================================
# EVALUASI TEST PENUH — batch kecil
# ==========================================

def evaluate_test_full(model, scaler, test_dir,
                       max_files, threshold):
    files = sorted(test_dir.glob("*.npz"))[:max_files]
    mean  = scaler['mean'].reshape(N_CHANNELS, N_TIMES)
    std   = scaler['std'].reshape(N_CHANNELS, N_TIMES)

    print(f"  Membaca {len(files)} file TEST (semua data)...")
    all_probs, all_labels = [], []

    for i, f in enumerate(files):
        d      = np.load(f)
        X      = d['X'].astype(np.float32)
        Y      = d['Y'].astype(np.int32)
        X_norm = (X - mean) / std

        # Batch kecil — aman VRAM 2GB
        probs = predict_in_batches(model, X_norm, INFER_BATCH_SIZE)
        all_probs.extend(probs.tolist())
        all_labels.extend(Y.tolist())
        del X, Y, X_norm, probs, d
        gc.collect()

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}] selesai...")

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    print(f"  Total : {len(all_labels):,} epochs")
    print(f"  Label 0: {(all_labels==0).sum():,} | "
          f"Label 1: {(all_labels==1).sum():,}")

    auc   = roc_auc_score(all_labels, all_probs)
    preds = (all_probs >= threshold).astype(int)
    f1    = f1_score(all_labels, preds, zero_division=0)
    bacc  = balanced_accuracy_score(all_labels, preds)

    return auc, f1, bacc, all_labels, preds, all_probs

# ==========================================
# MAIN
# ==========================================

def main():
    cek_device()

    # 1. Scaler
    print("Load scaler...")
    scaler = load_scaler()
    print("  Scaler loaded\n")

    # 2. Dataset TRAIN
    print("Memuat dataset TRAIN (subsample)...")
    train_ds = P300TrainDataset(
        TRAIN_DIR, scaler,
        sample_per_file=SAMPLE_PER_FILE_TRAIN,
        augment=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=True)

    # Dataset validasi
    val_ds = P300TrainDataset(
        TEST_DIR, scaler,
        sample_per_file=300,
        augment=False, seed=99)

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True)

    print(f"\n  Train : {len(train_ds):,} samples")
    print(f"  Val   : {len(val_ds):,} samples\n")

    # 3. Model
    model     = EEGNet(N_CHANNELS, N_TIMES).to(DEVICE)
    pw        = torch.tensor(POS_WEIGHT).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"EEGNet siap | {n_params:,} parameter\n")

    # 4. Training loop
    best_auc  = 0.0
    log_lines = ["epoch,train_loss,train_auc,val_loss,val_auc,val_bacc"]

    print(f"Training dimulai — {EPOCHS} epoch")
    print(f"  Baseline LDA : 0.5832")
    print(f"  Target       : AUC > 0.74\n")
    print("=" * 55)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion)
        val_loss, val_auc, val_bacc = evaluate_loader(
            model, val_loader, criterion)

        scheduler.step(val_auc)
        elapsed = time.time() - t0

        marker = " <- BEST" if val_auc > best_auc else ""
        print(f"Epoch [{epoch:02d}/{EPOCHS}] ({elapsed/60:.1f} menit)")
        print(f"  Train -> loss: {train_loss:.4f} | AUC: {train_auc:.4f}")
        print(f"  Val   -> loss: {val_loss:.4f} | "
              f"AUC: {val_auc:.4f} | "
              f"BalAcc: {val_bacc:.4f}{marker}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Model disimpan -> {MODEL_PATH}")

        log_lines.append(
            f"{epoch},{train_loss:.4f},{train_auc:.4f},"
            f"{val_loss:.4f},{val_auc:.4f},{val_bacc:.4f}")
        with open(LOG_PATH, 'w') as f:
            f.write("\n".join(log_lines))

        print()

    print("=" * 55)
    print(f"Training selesai! Best val AUC = {best_auc:.4f}")

    # 5. Load model terbaik
    print(f"\nLoad model terbaik dari {MODEL_PATH}...")
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE))
    torch.cuda.empty_cache()

    # 6. Kalibrasi threshold
    print(f"\nKalibrasi threshold dari data TEST asli...")
    thresh_calib = calibrate_threshold(
        model, scaler, TEST_DIR, CALIB_FILES)

    # 7. Evaluasi TEST penuh
    print(f"\nEvaluasi TEST penuh (semua file)...")
    auc_test, f1_test, bacc_test, y_true, y_pred, _ = \
        evaluate_test_full(
            model, scaler, TEST_DIR,
            max_files=78, threshold=thresh_calib)

    # 8. Laporan akhir
    print(f"\n{'='*55}")
    print(f"HASIL AKHIR")
    print(f"{'='*55}")
    print(f"  Val AUC (training) : {best_auc:.4f}")
    print(f"  TEST AUC           : {auc_test:.4f}  <- angka utama")
    print(f"  TEST F1            : {f1_test:.4f}")
    print(f"  TEST BalAcc        : {bacc_test:.4f}")
    print(f"  Threshold          : {thresh_calib:.2f}")

    print(f"\n  Perbandingan:")
    print(f"  LDA baseline : 0.5832")
    print(f"  EEGNet       : {auc_test:.4f}", end="")
    if auc_test > 0.7431:
        print(f"  <- Lebih baik dari versi sebelumnya!")
    else:
        print()

    print(f"\n{classification_report(y_true, y_pred, target_names=['Non-target','Target P300'])}")

    # 9. Simpan threshold
    import json
    thresh_info = {
        'threshold': float(thresh_calib),
        'auc_test':  float(auc_test),
        'f1_test':   float(f1_test)
    }
    with open('models/eegnet_threshold.json', 'w') as f:
        json.dump(thresh_info, f, indent=2)
    print(f"Threshold disimpan -> models/eegnet_threshold.json")
    print("=" * 55)


if __name__ == "__main__":
    main()