# import path so "pmas" can be found when running streamlit
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# webui/streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
from io import BytesIO
from pmas.main import run_full_pipeline

st.set_page_config(page_title="P-MAS Demo (Local, No-API)", layout="wide")
st.title("P-MAS — Local Multi-Agent Demo (Fairness + SHAP)")

st.markdown("""
Local demo (Pima Diabetes). Run baseline and mitigation methods and compare fairness metrics.
- **Reweight** = simple sample-weight rebalancing
- **ExpGrad** = Fairlearn ExponentiatedGradient (reduction)
""")

# Controls
col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    run_baseline = st.button("Run baseline")
with col_b:
    run_reweight = st.button("Run reweight mitigation")
with col_c:
    run_expgrad = st.button("Run expgrad mitigation")

st.markdown("### Or run all and compare")
run_all = st.button("Run all (baseline → reweight → expgrad)")

# Helpers
def metrics_dict_to_series(metrics):
    return pd.Series(metrics)

def group_metrics_to_frame(group_metrics):
    rows = []
    for metric, groups in group_metrics.items():
        for g, v in groups.items():
            rows.append({"metric": metric, "group": g, "value": float(v)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="metric", columns="group", values="value")
    pivot.columns = [f"group_{c}" for c in pivot.columns]
    return pivot

def export_csv(overall_df, group_df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        overall_df.to_excel(writer, sheet_name="Overall_Metrics")
        group_df.to_excel(writer, sheet_name="Group_Metrics")
    return buf.getvalue()

# Run pipelines
if run_baseline:
    with st.spinner("Running baseline..."):
        s = run_full_pipeline(use_mitigation=False)
    st.success("Baseline finished")
    st.subheader("Baseline overall metrics")
    st.json(s.get("eval_metrics"))
    st.subheader("Baseline group metrics")
    st.json(s.get("group_metrics"))
    shap_path = s.get("shap_path")
    if shap_path and Path(shap_path).exists():
        st.image(shap_path, use_column_width=True)

if run_reweight:
    with st.spinner("Running reweight mitigation..."):
        s = run_full_pipeline(use_mitigation=True, mitigation_method="reweight")
    st.success("Reweight finished")
    st.subheader("Reweight overall metrics")
    st.json(s.get("eval_metrics"))
    st.subheader("Reweight group metrics")
    st.json(s.get("group_metrics"))
    shap_path = s.get("shap_path")
    if shap_path and Path(shap_path).exists():
        st.image(shap_path, use_column_width=True)

if run_expgrad:
    with st.spinner("Running ExponentiatedGradient mitigation..."):
        s = run_full_pipeline(use_mitigation=True, mitigation_method="expgrad")
    st.success("ExpGrad finished")
    st.subheader("ExpGrad overall metrics")
    st.json(s.get("eval_metrics"))
    st.subheader("ExpGrad group metrics")
    st.json(s.get("group_metrics"))

    model = s.get("model")
    if isinstance(model, dict) and "clf" in model:
        clf = model["clf"]
        if clf.__class__.__name__ == "ExponentiatedGradient":
            top_info = None
            if hasattr(clf, "weights_") and hasattr(clf, "predictors_"):
                try:
                    weights = np.array(clf.weights_)
                    idx = int(np.argmax(weights))
                    w = float(weights[idx])
                    top_info = (idx, w)
                except Exception:
                    top_info = None
            elif hasattr(clf, "best_classifier_"):
                top_info = ("best_classifier_", None)
            if top_info:
                if top_info[1] is not None:
                    st.info(f"For ExponentiatedGradient, SHAP is computed on top-weighted predictor #{top_info[0]} (weight={top_info[1]:.3f})")
                else:
                    st.info("For ExponentiatedGradient, SHAP is computed on best_classifier_ (no weights available).")

    shap_path = s.get("shap_path")
    if shap_path and Path(shap_path).exists():
        st.image(shap_path, use_column_width=True)

# Run all
if run_all:
    with st.spinner("Running baseline..."):
        baseline = run_full_pipeline(use_mitigation=False)
    with st.spinner("Running reweight..."):
        reweight = run_full_pipeline(use_mitigation=True, mitigation_method="reweight")
    with st.spinner("Running expgrad..."):
        expgrad = run_full_pipeline(use_mitigation=True, mitigation_method="expgrad")

    st.success("All runs finished — comparison below")

    # Overall metrics
    baseline_metrics = baseline.get("eval_metrics", {})
    reweight_metrics = reweight.get("eval_metrics", {})
    expgrad_metrics = expgrad.get("eval_metrics", {})

    df_overall = pd.DataFrame({
        "baseline": metrics_dict_to_series(baseline_metrics),
        "reweight": metrics_dict_to_series(reweight_metrics),
        "expgrad": metrics_dict_to_series(expgrad_metrics)
    })
    st.subheader("Overall metrics (baseline vs reweight vs expgrad)")
    st.table(df_overall)

    # Group metrics
    def build_group_table(b, r, e):
        bdf = group_metrics_to_frame(b)
        rdf = group_metrics_to_frame(r)
        edf = group_metrics_to_frame(e)
        combined = pd.concat([bdf.add_suffix("_baseline"),
                              rdf.add_suffix("_reweight"),
                              edf.add_suffix("_expgrad")], axis=1)
        return combined

    group_combined = build_group_table(
        baseline.get("group_metrics", {}),
        reweight.get("group_metrics", {}),
        expgrad.get("group_metrics", {})
    )
    st.subheader("Group metrics comparison (metrics × group)")
    st.dataframe(group_combined.fillna("n/a"))

    # Legend explaining fairness metrics
    with st.expander("ℹ️ What do these group metrics mean?"):
        st.markdown("""
        - **Accuracy (per group):** % of correct predictions for each age group.  
        - **FPR (False Positive Rate):** % of non-diabetic patients incorrectly predicted as diabetic.  
        - **FNR (False Negative Rate):** % of diabetic patients incorrectly predicted as healthy.  

        Large gaps between groups indicate unfairness.  
        Mitigation methods aim to **reduce these disparities**, even if overall accuracy shifts slightly.
        """)

    # Download results
    csv_data = export_csv(df_overall, group_combined)
    st.download_button("⬇️ Download metrics as Excel", data=csv_data,
                       file_name="fairness_metrics_comparison.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # SHAP plots side-by-side
    row1, row2 = st.columns(2)
    with row1:
        shap_b = baseline.get("shap_path")
        if shap_b and Path(shap_b).exists():
            st.write("Baseline SHAP")
            st.image(shap_b, use_column_width=True)
    with row2:
        shap_e = expgrad.get("shap_path")
        if shap_e and Path(shap_e).exists():
            st.write("ExpGrad SHAP (top-weighted predictor)")
            st.image(shap_e, use_column_width=True)

    # Info about expgrad predictor weight
    model = expgrad.get("model")
    if isinstance(model, dict) and "clf" in model:
        clf = model["clf"]
        if clf.__class__.__name__ == "ExponentiatedGradient":
            top_info = None
            if hasattr(clf, "weights_") and hasattr(clf, "predictors_"):
                try:
                    weights = np.array(clf.weights_)
                    idx = int(np.argmax(weights))
                    w = float(weights[idx])
                    top_info = (idx, w)
                except Exception:
                    top_info = None
            elif hasattr(clf, "best_classifier_"):
                top_info = ("best_classifier_", None)
            if top_info:
                if top_info[1] is not None:
                    st.markdown(f"**Note:** For ExponentiatedGradient, SHAP is computed on top-weighted predictor **#{top_info[0]}** (weight={top_info[1]:.3f}).")
                else:
                    st.markdown("**Note:** For ExponentiatedGradient, SHAP is computed on `best_classifier_` (no weights available).")

st.markdown("---")
st.write("Tips:")
st.write("- Use the 'Run all' button to produce a side-by-side comparison for your demo video.")
st.write("- All computation runs locally using free OSS libraries.")
