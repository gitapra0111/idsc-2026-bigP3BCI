# ==========================================
# 📊 ULTIMATE COMPARISON: ROC CURVE PLOTTER
# (ANTI-OOM / RAM SAFE VERSION)
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
import joblib, torch, gc, os
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import torch.nn as nn
from tqdm import tqdm

# ==========================================
# CONFIGURATION & PATHS
# ==========================================

TEST_DIR           = Path('data/processed/dataset_npz/TEST')
SCALER_PATH        = Path('models/normalisasi/scaler.pkl')
MODEL_PATH_EEGNET  = Path('models/DL-EEGnet/eegnet_best.pt')
MODEL_PATH_XGB     = Path('models/ML-XGB/riemannian_xgb.pkl')
MODEL_PATH_CAT     = Path('models/ML-catboost/riemannian_catboost.pkl')

INFER_BATCH_SIZE = 256
N_CHANNELS       = 8
N_TIMES          = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# CLASS DEFINITIONS: EEGNet
# ==========================================

class EEGNet(nn.Module):
    def __init__(self, n_channels=8, n_times=128, F1=16, D=2, F2=32, dropout=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)), nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)), nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat = self.block3(self.block2(self.block1(dummy))).view(1, -1).shape[1]
        self.classifier = nn.Linear(flat, 1)

    def forward(self, x):
        return self.classifier(self.block3(self.block2(self.block1(x))).view(x.size(0), -1)).squeeze(1)

# ==========================================
# HELPER FUNCTIONS: INFERENCE
# ==========================================

@torch.no_grad()
def get_probs_eegnet(model, X_norm, batch_size=256):
    model.eval()
    all_probs = []
    for start in range(0, len(X_norm), batch_size):
        end = min(start + batch_size, len(X_norm))
        X_b = torch.tensor(X_norm[start:end], dtype=torch.float32).unsqueeze(1).to(DEVICE)
        prob = torch.sigmoid(model(X_b)).cpu().numpy()
        all_probs.extend(prob.tolist())
    return np.array(all_probs)

def get_probs_ml(pipeline, X_win, batch_size=2000):
    all_probs = []
    for start in range(0, len(X_win), batch_size):
        end = min(start + batch_size, len(X_win))
        prob = pipeline.predict_proba(X_win[start:end])[:, 1]
        all_probs.extend(prob.tolist())
    return np.array(all_probs)

# ==========================================
# MAIN PLOTTER (CHUNK PROCESSING - ANTI OOM)
# ==========================================

def main():
    print("=" * 65)
    print("      🚀 ULTIMATE COMPARISON: EEGNet vs XGBoost vs CatBoost")
    print("=" * 65)

    # 1. Load Scaler & Models
    print("\n📦 Loading Models & Scaler...")
    scaler_dict = joblib.load(SCALER_PATH)
    mean = scaler_dict['mean'].reshape(N_CHANNELS, N_TIMES)
    std  = scaler_dict['std'].reshape(N_CHANNELS, N_TIMES)

    model_dl = EEGNet(N_CHANNELS, N_TIMES).to(DEVICE)
    model_dl.load_state_dict(torch.load(MODEL_PATH_EEGNET, map_location=DEVICE))
    
    pipeline_xgb = joblib.load(MODEL_PATH_XGB)['pipeline']
    pipeline_cat = joblib.load(MODEL_PATH_CAT)['pipeline']

    # 2. Process Files in Chunks (Anti-OOM)
    files_test = sorted(TEST_DIR.glob("*.npz"))
    print(f"\n🔮 Processing {len(files_test)} TEST files using chunking system (Anti-OOM)...")

    y_true_all = []
    probs_eegnet_all = []
    probs_xgb_all = []
    probs_cat_all = []

    for f in tqdm(files_test, desc="Evaluating Files"):
        d = np.load(f)
        X = d['X'].astype(np.float32)
        Y = d['Y'].astype(np.int32)
        
        # Save Ground Truth Labels
        y_true_all.extend(Y.tolist())

        # A. EEGNet Inference (Requires Full Normalization)
        X_norm = (X - mean) / std
        prob_e = get_probs_eegnet(model_dl, X_norm, INFER_BATCH_SIZE)
        probs_eegnet_all.extend(prob_e.tolist())
        del X_norm

        # B. ML Models Inference (Requires Time Windowing 25-102)
        X_ml = X[:, :, 25:102]
        prob_x = get_probs_ml(pipeline_xgb, X_ml)
        prob_c = get_probs_ml(pipeline_cat, X_ml)
        
        probs_xgb_all.extend(prob_x.tolist())
        probs_cat_all.extend(prob_c.tolist())

        # Clear memory to free up RAM for the next file
        del X, Y, X_ml, prob_e, prob_x, prob_c, d
        gc.collect()

    y_true = np.array(y_true_all)
    print(f"\n✅ Done! A total of {len(y_true):,} predictions successfully collected without overloading RAM.")

    # 3. Calculate ROC Curve and AUC
    print("\n📊 Calculating ROC Curve & AUC...")
    fpr_eegnet, tpr_eegnet, _ = roc_curve(y_true, probs_eegnet_all)
    auc_eegnet = auc(fpr_eegnet, tpr_eegnet)
    
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, probs_xgb_all)
    auc_xgb = auc(fpr_xgb, tpr_xgb)

    fpr_cat, tpr_cat, _ = roc_curve(y_true, probs_cat_all)
    auc_cat = auc(fpr_cat, tpr_cat)

    print(f"\n{'='*65}")
    print(f"FINAL COMPARISON RESULTS")
    print(f"{'='*65}")
    print(f"  1. EEGNet (DL)         : AUC = {auc_eegnet:.4f} 🔥")
    print(f"  2. XGBoost (Riemannian): AUC = {auc_xgb:.4f}")
    print(f"  3. CatBoost (Riemannian): AUC = {auc_cat:.4f}")
    print(f"{'='*65}")

    # 4. Draw Professional Plot
    print("\n🖼️ Opening Plot Window...")
    plt.figure(figsize=(10, 8), dpi=100)
    
    plt.plot([0, 1], [0, 1], color='darkblue', lw=2, linestyle='--', label='Chance Level (AUC = 0.50)')
    
    plt.plot(fpr_eegnet, tpr_eegnet, color='crimson', lw=3, label=f'1. EEGNet (AUC = {auc_eegnet:.4f}) 🔥')
    plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=3, linestyle='-', label=f'2. XGBoost Opt (AUC = {auc_xgb:.4f})')
    plt.plot(fpr_cat, tpr_cat, color='forestgreen', lw=3, linestyle='-', label=f'3. CatBoost Rev (AUC = {auc_cat:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Incorrectly Predicted Non-Target)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Successfully Predicted Target P300)', fontsize=12, fontweight='bold')
    plt.title("Ultimate Comparison P300 Classification\n(EEGNet vs XGBoost vs CatBoost)", fontsize=16, fontweight='bold')
    
    plt.legend(loc="lower right", fontsize=11, shadow=True)
    plt.grid(True, which='both', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()