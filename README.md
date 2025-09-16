# FairLens — Interpretable Bias Detection (Pima + Adult)

**FairLens** is a local, reproducible demo that detects, quantifies and mitigates bias in classification models using **Fairlearn** and **SHAP**.  
It is implemented as a small multi-agent pipeline and visualized via a polished **Streamlit** UI.

---

## 🚀 Features

- **Datasets (both supported)**:
  - **Pima Diabetes** (UCI) — small, fast for quick demos
  - **Adult Income** (UCI) — larger, demographic fairness examples
- **Bias detection**: per-group metrics (accuracy, FPR, FNR)
- **Bias mitigation**:
  - **Reweighting** (sample rebalancing)
  - **ExponentiatedGradient** (Fairlearn reductions)
- **Interpretability**: SHAP explanations and SHAP summary plots
- **UI**: Streamlit dashboard with comparison tabs, KPI cards and fairness gaps
- **Reproducibility**: `generate_report.py` produces `report.ipynb` (static) and `report_executed.ipynb` (optional executed notebook)
- **Screenshots**: `tools/create_demo_screenshots.py` automates creation of demo PNGs you can embed in the README

---

## Quickstart (offline-capable)

1. Clone & setup
```bash
git clone https://github.com/<your-username>/FairLens.git
cd FairLens
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. (Recommended) Download Adult dataset for offline runs:
```bash
Copy code
mkdir -p data
curl -L -o data/adult.csv https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
# Pima will auto-download, or add data/pima_diabetes.csv if you prefer to cache locally.

3. Run Streamlit UI
```bash
Copy code
streamlit run webui/streamlit_app.py

Open http://localhost:8502 and use the sidebar to select Pima Diabetes or Adult Income. Toggle Use cache only for deterministic offline demos.

Project layout
.
├─ pmas/                      # agents, orchestrator, main pipeline
├─ webui/
│  └─ streamlit_app.py        # Streamlit polished UI
├─ generate_report.py         # builds report.ipynb (and executes)
├─ tools/
│  └─ create_demo_screenshots.py
├─ data/                      # cached datasets (adult.csv optional)
├─ outputs/                   # SHAP images, demo screenshots
├─ requirements.txt
└─ README.md

Motivation & usage in applications

This project demonstrates practical skills: fairness-aware modelling, interpretable ML (SHAP), packaging experiments into reproducible artifacts (Streamlit + notebook + screenshots). Use it in your CV / SOP by linking the repo and noting the datasets and techniques used.


