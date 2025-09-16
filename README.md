# FairLens — Interpretable Bias Detection (Pima + Adult)


# FairLens — Interpretable Bias Detection

**FairLens** is a fairness-aware, interpretable machine learning demo.  
It combines **multi-agent orchestration**, **Fairlearn** (bias mitigation), and **SHAP** (explainability) to detect, quantify, and mitigate bias across demographic groups.

---

## ✨ Features
- **Multi-agent pipeline**: modular agents for data loading, modeling, mitigation, evaluation, explanation.  
- **Datasets supported**:  
  - *Pima Indians Diabetes* (health prediction)  
  - *Adult Income* (census data)  
- **Bias mitigation methods**:  
  - Baseline (no mitigation)  
  - Reweighting (sample-weight rebalancing)  
  - ExponentiatedGradient (Fairlearn reductions)  
- **Explainability**: SHAP value plots for feature importance.  
- **Fairness metrics**: accuracy, precision, recall, FPR/FNR by group + fairness gaps.  
- **Interactive dashboard**: Streamlit UI with tabs, cards, and comparison tables.  
- **Static report**: auto-generated `report.ipynb` with reproducible analysis.

---

## 📸 Screenshots

### Streamlit Dashboard (Pima Diabetes)
![Demo Screenshot — Pima Diabetes](outputs/demo_pima.png)

### Streamlit Dashboard (Adult Income)
![Demo Screenshot — Adult Income](outputs/demo_adult.png)

---

## 🚀 Getting Started

### 1. Clone & set up
```bash
git clone https://github.com/<your-username>/FairLens-Multiagent-ML.git
cd FairLens-Multiagent-ML
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
2. Run Streamlit demo
bash
Copy code
streamlit run webui/streamlit_app.py
Choose dataset (Pima or Adult)

Run baseline / mitigation pipelines

Compare fairness metrics & SHAP plots

3. Generate static report
bash
Copy code
python generate_report.py
This creates report.ipynb with metrics and SHAP visualizations.
👉 To include live outputs, open the notebook in VS Code and Run All.

📊 Example Results
Overall metrics (Adult Income, cache-only):

metric	baseline	reweight	expgrad
accuracy	0.815	0.813	0.781
precision	0.701	0.756	0.629
recall	0.405	0.329	0.217

Fairness gap (FPR difference):

Baseline: 0.209

Reweight: 0.209

Expgrad: 0.047

🛠️ Tech Stack
Python 3.12

Streamlit — interactive dashboard

Fairlearn — fairness metrics & reductions

SHAP — model interpretability

Scikit-learn / Pandas / Matplotlib

📂 Repo Structure
arduino
Copy code
pmas/
  agents/              # data, model, mitigation, explainability agents
  main.py              # orchestrates full pipeline
  orchestrator.py
webui/
  streamlit_app.py     # polished dashboard
tools/
  create_demo_screenshots.py
outputs/
  demo_pima.png
  demo_adult.png
report.ipynb
requirements.txt
README.md
💡 Motivation
Bias in ML models can have serious social consequences.
This project demonstrates a practical, interpretable pipeline for bias detection & mitigation — useful for learning fairness concepts and for showcasing reproducible ML research.

📜 License
MIT License. Free to use and adapt.

👩‍💻 Built by Samriddhi Sharma — fairness, interpretability, and ML systems.