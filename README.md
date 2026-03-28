
# 🧠 SIGSQUAD: P300 Brain-Computer Interface (BCI) Decoder
**Data Science Competition - IDSC 2026**

This repository contains the complete pipeline for data processing, model training, and a UI/UX prototype for decoding P300 brainwave signals from the **bigP3BCI** dataset. This project aims to serve as an assistive technology for patients with *Locked-in Syndrome*.

---

## 📄 Technical Report
The comprehensive documentation detailing our methodology, data processing, model architecture, and evaluation metrics can be accessed here:
📥 **[Download or Read the Full Tech Report Here](https://raw.githubusercontent.com/gitapra0111/idsc-2026-bigP3BCI/main/tech-report/Sigsquad-Tech_Report_IDSC.pdf)**


## 🚀 Live Demo (UI/UX Prototype)
We translated the sharpness of our model predictions into an intuitive, ready-to-use Virtual Keyboard prototype.
🔗 🔗 **[Access the P300 Speller Prototype Here](https://idsc-2026-big-p3-bci.vercel.app/show/speller.html)**

---

## 🔬 Methodology & Models
This project compares traditional Machine Learning approaches based on feature extraction with an end-to-end Deep Learning approach:
1. **EEGNet (Deep Learning):** Our primary model. It is lightweight, efficient, and capable of extracting spatial-temporal patterns directly from raw EEG signals without manual feature engineering.
2. **XGBoost & CatBoost (Machine Learning):** Baseline models optimized using *Riemannian Geometry* for Covariance Matrix feature extraction.

## 📊 Results
Based on extensive evaluation on a massive TEST dataset using the **AUC (Area Under the ROC Curve)** metric to handle extreme class imbalance:
* **EEGNet** outperformed traditional ML models with an **AUC score of ~0.7300**, demonstrating superior sensitivity in capturing the rare P300 signal spikes compared to the baselines.

---

## 📁 Main Repository Structure
* `tech-report/` : Contains the comprehensive Technical Report document for IDSC 2026.
* `src/preprocessing/` : Scripts for data cleaning, band-pass filtering, epoching, and normalization.
* `src/training/` : AI model training scripts (EEGNet, CatBoost, XGBoost).
* `src/visual/` : ROC curve comparison and visualization scripts (Anti-OOM optimized).
* `models/` : Trained model weights (`.pt` and `.pkl`) along with the Scaler files.
* `notebooks/` : Exploratory Data Analysis (EDA) and initial preprocessing experiments.
* `show/` & `index.html` : Frontend source code for the P300 Speller UI/UX prototype.

---
*Built with ☕ and 💻 by Team SIGSQUAD.*