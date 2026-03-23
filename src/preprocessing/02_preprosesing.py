# ==========================================
# 🚀 FINAL PREPROCESSING (READY TO RUN)
# ==========================================

import gc
import mne
import numpy as np
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================

RAW_DIR = Path('data/raw/bigp3bci/bigP3BCI-data')
OUT_DIR = Path('data/processed/dataset_npz')

TRAIN_DIR = OUT_DIR / 'TRAIN'
TEST_DIR = OUT_DIR / 'TEST'

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 50
SFREQ_TARGET = 128

CH_P300 = [
    'EEG_Fz', 'EEG_Cz', 'EEG_Pz',
    'EEG_P3', 'EEG_P4',
    'EEG_PO7', 'EEG_PO8', 'EEG_Oz'
]

# ==========================================
# SAVE FUNCTION
# ==========================================

def save_batch(X_list, Y_list, S_list, SE_list, part, folder, name):

    if len(X_list) == 0:
        return part

    print(f"📦 Saving {name} part {part}")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    Y = np.concatenate(Y_list, axis=0)
    S = np.concatenate(S_list, axis=0)
    SE = np.concatenate(SE_list, axis=0)

    np.savez_compressed(
        folder / f"{name}_part{part}.npz",
        X=X,
        Y=Y,
        subject=S,
        session=SE
    )

    X_list.clear()
    Y_list.clear()
    S_list.clear()
    SE_list.clear()

    gc.collect()

    return part + 1

# ==========================================
# PROCESS FILE
# ==========================================

def process_file(file_path):

    # ===== METADATA =====
    parts = file_path.parts

    try:
        study   = parts[-6]
        subject = parts[-5]
        session = parts[-4]
        split   = parts[-3]
    except:
        raise ValueError("Invalid folder structure")

    subject_id = int(subject.split('_')[1])
    session_id = int(session.replace('SE', ''))

    # ===== LOAD =====
    raw = mne.io.read_raw_edf(file_path.as_posix(), preload=True, verbose=False)

    # ===== VALIDASI CHANNEL =====
    available = raw.ch_names
    missing = [ch for ch in CH_P300 if ch not in available]
    if len(missing) > 0:
        raise ValueError(f"Missing channel: {missing}")

    # ===== EVENTS =====
    events = mne.find_events(raw, stim_channel='StimulusBegin', verbose=False)
    if len(events) == 0:
        raise ValueError("No events")

    # ===== FILTER =====
    raw.filter(0.1, 30.0, picks=CH_P300, verbose=False)

    # ===== EPOCH =====
    epochs = mne.Epochs(
        raw,
        events,
        picks=CH_P300,
        tmin=-0.2,
        tmax=0.8,
        baseline=(None, 0),
        preload=True,
        verbose=False
    )

    if len(epochs) == 0:
        raise ValueError("Empty epochs")

    # ===== LABEL FIX (KRUSIAL) =====
    stim = raw.get_data(picks=['StimulusType'])[0]
    labels = stim[epochs.events[:, 0]].astype(int)

    # ===== RESAMPLE =====
    epochs.resample(SFREQ_TARGET)

    X = epochs.get_data(copy=False)

    # ===== METADATA ARRAY =====
    subject_array = np.full(len(labels), subject_id, dtype=np.int16)
    session_array = np.full(len(labels), session_id, dtype=np.int16)

    del raw, epochs

    return X, labels, subject_array, session_array, split.lower()

# ==========================================
# MAIN PIPELINE
# ==========================================

def main():

    # 🔥 FIX: SORT FILE (WAJIB)
    files = sorted(RAW_DIR.rglob("*.edf"))

    print(f"📁 Total file: {len(files)}")

    train_X, train_Y, train_S, train_SE = [], [], [], []
    test_X, test_Y, test_S, test_SE = [], [], [], []

    part_train, part_test = 1, 1

    for i, file in enumerate(files):

        print(f"[{i+1}/{len(files)}] {file.name}")

        try:
            X, Y, S, SE, split = process_file(file)

            # 🔥 FIX: SPLIT CLEAN
            if split == "train":
                train_X.append(X)
                train_Y.append(Y)
                train_S.append(S)
                train_SE.append(SE)

            elif split == "test":
                test_X.append(X)
                test_Y.append(Y)
                test_S.append(S)
                test_SE.append(SE)

            # memory cleanup
            del X, Y, S, SE

        except Exception as e:
            print(f"❌ Skip {file.name} | {e}")
            continue

        # ===== SAVE BATCH =====
        if len(train_X) >= BATCH_SIZE:
            part_train = save_batch(
                train_X, train_Y, train_S, train_SE,
                part_train, TRAIN_DIR, "TRAIN"
            )

        if len(test_X) >= BATCH_SIZE:
            part_test = save_batch(
                test_X, test_Y, test_S, test_SE,
                part_test, TEST_DIR, "TEST"
            )

    # ===== SAVE SISA =====
    print("📦 Saving remaining data...")

    part_train = save_batch(
        train_X, train_Y, train_S, train_SE,
        part_train, TRAIN_DIR, "TRAIN"
    )

    part_test = save_batch(
        test_X, test_Y, test_S, test_SE,
        part_test, TEST_DIR, "TEST"
    )

    print("🎉 DONE (SIAP TRAINING TANPA LEAKAGE)")

# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    main()

    # clean up memory