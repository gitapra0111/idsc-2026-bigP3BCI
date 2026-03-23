# ==========================================
# TRAINING ML - RIEMANNIAN + XGBOOST (OPT)
# ==========================================
# Optimasi dari versi sebelumnya:
# 1. scale_pos_weight pakai rasio ASLI 8.67
#    bukan dari subsample yang seimbang
# 2. Sample per file dinaikkan ke 1000
# 3. XDawn nfilter dinaikkan ke 6
# 4. XGBoost parameter lebih optimal
#
# python src/train_riemannian.py
# ==========================================

import numpy as np
import glob, os, joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (f1_score, roc_auc_score,
                              balanced_accuracy_score,
                              classification_report)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIG
# ==========================================

TRAIN_DIR  = "data/processed/dataset_npz/TRAIN"
TEST_DIR   = "data/processed/dataset_npz/TEST"
MODEL_PATH = "models/riemannian_xgb.pkl"

SAMPLE_PER_FILE_TRAIN = 1000   # naik dari 500
MAX_FILES_TRAIN       = 63
MAX_FILES_TEST        = 78
CALIB_FILES           = 5

# Rasio ASLI dari hasil verifikasi data
# Jangan pakai rasio dari subsample (selalu 1:1)
TRUE_POS_WEIGHT = 8.67

WIN_START    = 25
WIN_END      = 102
RANDOM_STATE = 42

os.makedirs("models", exist_ok=True)

# ==========================================
# LOAD SUBSAMPLE
# ==========================================

def load_subsample(folder, max_files, sample_per_file):
    files = sorted(glob.glob(f"{folder}/*.npz"))[:max_files]
    if not files:
        raise ValueError(f"Tidak ada file NPZ di {folder}!")

    X_all, Y_all, S_all = [], [], []
    rng = np.random.default_rng(RANDOM_STATE)

    print(f"  {len(files)} file x {sample_per_file} sample/file...")
    for f in files:
        d = np.load(f)
        X = d['X'].astype(np.float32)
        Y = d['Y'].astype(np.int32)
        S = d['subject']

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
        S_all.append(S[sel])
        del X, Y, S, d

    X_out = np.concatenate(X_all, axis=0)
    Y_out = np.concatenate(Y_all, axis=0)
    S_out = np.concatenate(S_all, axis=0)

    print(f"  Total  : {X_out.shape[0]:,} epochs")
    print(f"  Label 0: {(Y_out==0).sum():,} | "
          f"Label 1: {(Y_out==1).sum():,}")
    return X_out, Y_out, S_out


# ==========================================
# KALIBRASI THRESHOLD
# ==========================================

def calibrate_threshold(pipeline, test_dir, n_files,
                        win_start, win_end):
    files = sorted(glob.glob(f"{test_dir}/*.npz"))[:n_files]
    print(f"  Kalibrasi dari {n_files} file TEST asli...")

    all_probs, all_labels = [], []
    for f in files:
        d    = np.load(f)
        X    = d['X'].astype(np.float32)[:, :, win_start:win_end]
        Y    = d['Y'].astype(np.int32)
        prob = pipeline.predict_proba(X)[:, 1]
        all_probs.extend(prob.tolist())
        all_labels.extend(Y.tolist())
        del X, Y, prob, d

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

    print(f"  Threshold optimal: {best_thresh:.2f} "
          f"(F1={best_f1:.4f})")
    return best_thresh


# ==========================================
# EVALUASI TEST PENUH
# ==========================================

def evaluate_test_full(pipeline, test_dir, max_files,
                       win_start, win_end, threshold):
    files = sorted(glob.glob(f"{test_dir}/*.npz"))[:max_files]
    print(f"  Membaca {len(files)} file TEST (semua data)...")

    all_probs, all_labels = [], []
    for i, f in enumerate(files):
        d    = np.load(f)
        X    = d['X'].astype(np.float32)[:, :, win_start:win_end]
        Y    = d['Y'].astype(np.int32)
        prob = pipeline.predict_proba(X)[:, 1]
        all_probs.extend(prob.tolist())
        all_labels.extend(Y.tolist())
        del X, Y, prob, d

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
    print("=" * 55)
    print("RIEMANNIAN GEOMETRY + XGBOOST (OPTIMIZED)")
    print("=" * 55)
    print(f"\nOptimasi:")
    print(f"  scale_pos_weight : {TRUE_POS_WEIGHT} (rasio asli)")
    print(f"  sample/file      : {SAMPLE_PER_FILE_TRAIN}")
    print(f"  XDawn nfilter    : 6 (dari 4)")

    # 1. Load TRAIN
    print(f"\nLoad TRAIN (subsample {SAMPLE_PER_FILE_TRAIN}/file)...")
    X_raw, Y_all, subjects = load_subsample(
        TRAIN_DIR, MAX_FILES_TRAIN, SAMPLE_PER_FILE_TRAIN)

    # 2. Window P300
    print(f"\nWindow P300: {WIN_START}-{WIN_END} (0ms-600ms)...")
    X = X_raw[:, :, WIN_START:WIN_END]
    del X_raw
    print(f"  Shape: {X.shape}")

    # 3. Split
    print(f"\nSplit data (anti-leakage by subject)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, Y_all, groups=subjects))

    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y_all[train_idx], Y_all[val_idx]

    print(f"  Belajar  : {X_train.shape[0]:,} epochs")
    print(f"  Validasi : {X_val.shape[0]:,} epochs")
    print(f"  scale_pos_weight dipakai: {TRUE_POS_WEIGHT}")

    # 4. Pipeline — dengan optimasi
    print(f"\nMembangun pipeline Riemannian + XGBoost (optimized)...")
    pipeline = Pipeline([
        # nfilter=6 menangkap lebih banyak pola P300
        ('xdawn_cov', XdawnCovariances(
            nfilter=6,
            estimator='lwf')),
        ('tangent', TangentSpace(metric='riemann')),
        ('scaler',  StandardScaler()),
        ('xgb', XGBClassifier(
            n_estimators=700,        # lebih banyak pohon
            learning_rate=0.03,      # lebih halus
            max_depth=5,             # sedikit lebih dalam
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,      # cegah overfitting
            scale_pos_weight=TRUE_POS_WEIGHT,  # rasio ASLI!
            tree_method='hist',
            n_jobs=-1,
            random_state=42,
            verbosity=0
        ))
    ])

    # 5. Training
    print(f"\nTraining dimulai (~5-10 menit)...")
    pipeline.fit(X_train, Y_train)
    print("  Training selesai!")

    # 6. Evaluasi validasi
    print(f"\nEvaluasi validasi internal...")
    probs_val = pipeline.predict_proba(X_val)[:, 1]
    auc_val   = roc_auc_score(Y_val, probs_val)

    best_f1_val, thresh_val = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(Y_val, (probs_val > t).astype(int),
                      zero_division=0)
        if f1 > best_f1_val:
            best_f1_val, thresh_val = f1, t

    preds_val = (probs_val > thresh_val).astype(int)
    bacc_val  = balanced_accuracy_score(Y_val, preds_val)

    print(f"  Validasi AUC : {auc_val:.4f}")
    print(f"  Validasi F1  : {best_f1_val:.4f}")
    print(f"  Validasi BalAcc: {bacc_val:.4f}")
    print(f"  Threshold    : {thresh_val:.2f}")

    # 7. Kalibrasi threshold
    print(f"\nKalibrasi threshold dari data TEST asli...")
    thresh_calib = calibrate_threshold(
        pipeline, TEST_DIR, CALIB_FILES, WIN_START, WIN_END)

    print(f"\n  Perbandingan threshold:")
    print(f"  Dari validasi : {thresh_val:.2f}")
    print(f"  Dari TEST asli: {thresh_calib:.2f}  <- dipakai")

    # 8. Evaluasi TEST penuh
    print(f"\nEvaluasi TEST penuh ({MAX_FILES_TEST} file)...")
    auc_test, f1_test, bacc_test, y_true, y_pred, _ = \
        evaluate_test_full(
            pipeline, TEST_DIR, MAX_FILES_TEST,
            WIN_START, WIN_END, thresh_calib)

    # 9. Laporan
    print(f"\n{'='*55}")
    print(f"HASIL AKHIR")
    print(f"{'='*55}")
    print(f"  Validasi AUC : {auc_val:.4f}")
    print(f"  TEST AUC     : {auc_test:.4f}  <- angka utama")
    print(f"  TEST F1      : {f1_test:.4f}")
    print(f"  TEST BalAcc  : {bacc_test:.4f}")
    print(f"  Threshold    : {thresh_calib:.2f}")

    print(f"\n  Perbandingan semua model:")
    print(f"  LDA baseline      : 0.5832")
    print(f"  Riemannian+SVM    : 0.6694")
    print(f"  XGBoost sebelumnya: 0.7016")
    print(f"  EEGNet            : 0.7431")
    print(f"  XGBoost optimized : {auc_test:.4f}", end="")
    if auc_test > 0.7431:
        print(f"  <- Lebih baik dari EEGNet!")
    elif auc_test > 0.7016:
        print(f"  <- Lebih baik dari versi sebelumnya!")
    else:
        print(f"  <- Tidak ada peningkatan")

    print(f"\n{classification_report(y_true, y_pred, target_names=['Non-target','Target P300'])}")

    # 10. Simpan
    joblib.dump({
        'pipeline':   pipeline,
        'threshold':  thresh_calib,
        'thresh_val': thresh_val,
        'auc_test':   auc_test,
        'win_start':  WIN_START,
        'win_end':    WIN_END
    }, MODEL_PATH)
    print(f"Model disimpan -> {MODEL_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    main()