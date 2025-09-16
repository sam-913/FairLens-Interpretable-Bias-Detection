# FairLens — Interpretable Bias Detection


FairLens is a **fairness-aware ML demo** that combines:
- **Bias Detection** — Quantify disparities in accuracy, false positive/negative rates across groups  
- **Bias Mitigation** — Apply methods like **Reweighting** and **Exponentiated Gradient (Fairlearn)**  
- **Interpretability** — Use **SHAP explainability** to visualize feature importance  

Built as a **multi-agent pipeline** with a polished **Streamlit UI**, it highlights responsible AI practices for healthcare and income prediction datasets.

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

## 🌐 Demo (Streamlit)

## 🚀 Live Demo
👉 [Try the Streamlit app here]([https://your-streamlit-app-link](https://fairlens.streamlit.app/))  

- Dataset toggle: **Pima Diabetes** (healthcare) and **Adult Income** (socioeconomic)  
- Compare **Baseline vs Mitigation** side-by-side  
- Visualize **SHAP explanations** and **fairness gaps** (e.g., FPR difference, accuracy difference)  

---

## 📊 Example Results

**Overall metrics (Baseline vs Reweight vs ExpGrad):**

| Metric     | Baseline | Reweight | ExpGrad |
|------------|----------|----------|---------|
| Accuracy   | 0.74     | 0.74     | 0.75    |
| Precision  | 0.67     | 0.67     | 0.71    |
| Recall     | 0.50     | 0.50     | 0.49    |

**Fairness gap (FPR difference):** reduced after mitigation.  

---

## 📂 Project Structure

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

---

## 🛠️ Tech Stack

- **Python 3.12**
- [scikit-learn](https://scikit-learn.org/) — ML models  
- [Fairlearn](https://fairlearn.org/) — Fairness metrics & reductions  
- [SHAP](https://shap.readthedocs.io/) — Interpretability  
- [Streamlit](https://streamlit.io/) — UI  

---

## 📘 Report
A reproducible notebook `report.ipynb` is included with:  
- End-to-end pipeline runs  
- SHAP plots inline  
- Fairness gap calculations  

---

## ✨ Why This Project?
FairLens demonstrates **responsible AI deployment**:
- Detecting and explaining **bias**  
- Applying **mitigation techniques**  
- Providing a **transparent, interactive UI**  

This is especially relevant for **AI in healthcare and socioeconomic decision-making**.

---

📜 License
MIT License. Free to use and adapt.

---

👩‍💻 Built by Samriddhi Sharma — fairness, interpretability, and ML systems.

---