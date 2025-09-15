# P-MAS Free Starter (Local-only) — Pima Diabetes Demo

This repository is a local-only demo of P-MAS (Private Multi-Agent Automation Stack).
It loads the Pima Indians Diabetes dataset, trains a logistic regression,
computes fairness metrics (Fairlearn), runs a simple reweighting mitigation,
and produces SHAP explanations — all locally with open-source libraries.

## Quickstart

1. Create virtualenv and activate
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
