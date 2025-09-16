# webui/streamlit_app.py
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import zipfile
from pmas.main import run_full_pipeline

# --- Page config ---
st.set_page_config(page_title="FairLens — Polished UI", layout="wide")
PRIMARY = "#0b5fff"
MUTED = "#6b7280"

# --- small CSS tweaks for polish ---
st.markdown(
    f"""
    <style>
      .topbar {{ height:6px; width:100%; background: linear-gradient(90deg, {PRIMARY}, #2db4ff); border-radius:8px; margin-bottom:18px;}}
      .title {{ font-size:26px; font-weight:700; }}
      .subtitle {{ color: {MUTED}; margin-top:4px; font-size:13px; }}
      .card {{ background: #fff; padding:16px; border-radius:10px; box-shadow: 0 6px 18px rgba(12,18,30,0.04); }}
      .kpi {{"display:flex; gap:8px; align-items:center;"}}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("<div class='topbar'></div>", unsafe_allow_html=True)
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("<div class='title'>FairLens</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Interpretable Bias Detection — Fairlearn + SHAP (local)</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align:right'><span class='subtitle'>Status: Local • Offline-capable</span></div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Settings")
dataset_choice = st.sidebar.selectbox("Dataset", ["Pima Diabetes", "Adult Income"])
use_cache_only = st.sidebar.checkbox("Use cache only (offline)", value=True, help="Use local cached files (e.g. data/adult.csv); avoids network.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Demo features**")
st.sidebar.markdown("- Baseline, Reweight, ExpGrad\n- SHAP explanations\n- Fairness gap metrics\n- Export CSV/ZIP")

# Buttons
st.write("")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### Actions")
b1, b2, b3, b4 = st.columns([1.2,1.2,1.2,2.4])
with b1:
    run_baseline = st.button("Run baseline")
with b2:
    run_reweight = st.button("Run Reweight")
with b3:
    run_expgrad = st.button("Run ExpGrad")
with b4:
    run_all = st.button("Run all (baseline → reweight → expgrad)")
st.markdown("</div>", unsafe_allow_html=True)

# helpers
def metrics_series(m):
    return pd.Series({k: float(v) for k, v in (m or {}).items()})

def group_frame(group_metrics):
    rows=[]
    for metric, groups in (group_metrics or {}).items():
        for g, v in groups.items():
            rows.append({"metric": metric, "group": int(g), "value": float(v)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="metric", columns="group", values="value")
    pivot.columns = [f"group_{c}" for c in pivot.columns]
    return pivot

def compute_gaps(group_metrics):
    try:
        acc_gap = abs(group_metrics["accuracy"][0] - group_metrics["accuracy"][1])
        fpr_gap = abs(group_metrics["fpr"][0] - group_metrics["fpr"][1])
        fnr_gap = abs(group_metrics["fnr"][0] - group_metrics["fnr"][1])
        return {"accuracy_gap": float(acc_gap), "fpr_gap": float(fpr_gap), "fnr_gap": float(fnr_gap)}
    except Exception:
        return {}

def color_for_gap(x):
    # threshold: green <0.05, amber <0.15, red >=0.15 (tune as needed)
    if x < 0.05: return "✅"
    if x < 0.15: return "⚠️"
    return "❗"

def export_zip(overall_df, group_df, gaps_df=None):
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("overall_metrics.csv", overall_df.to_csv(index=True).encode("utf-8"))
        z.writestr("group_metrics.csv", group_df.to_csv(index=True).encode("utf-8"))
        if gaps_df is not None:
            z.writestr("fairness_gaps.csv", gaps_df.to_csv(index=True).encode("utf-8"))
    buf.seek(0)
    return buf.getvalue()

# runner wrapper
def run_pipeline(use_mitigation=False, mitigation_method="reweight"):
    return run_full_pipeline(use_mitigation=use_mitigation,
                             mitigation_method=mitigation_method,
                             dataset=dataset_choice,
                             use_cache_only=use_cache_only)

# UI results area using tabs
tab_metrics, tab_groups, tab_shap, tab_about = st.tabs(["Metrics", "Group comparison", "SHAP", "About"])

# Run actions: baseline / reweight / expgrad / all
last_state = None
if run_baseline:
    with st.spinner("Running baseline..."):
        s = run_pipeline(use_mitigation=False)
    last_state = s
    st.success("Baseline finished")
    with tab_metrics:
        st.subheader("Baseline — overall metrics")
        m = s.get("eval_metrics", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
        c2.metric("Precision", f"{m.get('precision', 0):.3f}")
        c3.metric("Recall", f"{m.get('recall', 0):.3f}")
        st.markdown("**Fairness gaps**")
        gaps = compute_gaps(s.get("group_metrics", {}))
        if gaps:
            gap_df = pd.Series(gaps).to_frame("value")
            gap_df["flag"] = gap_df["value"].apply(color_for_gap)
            st.table(gap_df.style.format("{:.4f}"))
    with tab_groups:
        st.subheader("Group metrics (baseline)")
        st.dataframe(group_frame(s.get("group_metrics", {})).style.format("{:.4f}"))
    with tab_shap:
        st.subheader("Baseline SHAP")
        sp = s.get("shap_path")
        if sp and Path(sp).exists():
            st.image(sp, use_column_width=True)

if run_reweight:
    with st.spinner("Running reweight..."):
        s = run_pipeline(use_mitigation=True, mitigation_method="reweight")
    last_state = s
    st.success("Reweight finished")
    with tab_metrics:
        st.subheader("Reweight — overall metrics")
        m = s.get("eval_metrics", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
        c2.metric("Precision", f"{m.get('precision', 0):.3f}")
        c3.metric("Recall", f"{m.get('recall', 0):.3f}")
        st.markdown("**Fairness gaps**")
        gaps = compute_gaps(s.get("group_metrics", {}))
        if gaps:
            gap_df = pd.Series(gaps).to_frame("value")
            gap_df["flag"] = gap_df["value"].apply(color_for_gap)
            st.table(gap_df.style.format("{:.4f}"))
    with tab_groups:
        st.subheader("Group metrics (reweight)")
        st.dataframe(group_frame(s.get("group_metrics", {})).style.format("{:.4f}"))
    with tab_shap:
        st.subheader("Reweight SHAP")
        sp = s.get("shap_path")
        if sp and Path(sp).exists():
            st.image(sp, use_column_width=True)

if run_expgrad:
    with st.spinner("Running expgrad..."):
        s = run_pipeline(use_mitigation=True, mitigation_method="expgrad")
    last_state = s
    st.success("ExpGrad finished")
    with tab_metrics:
        st.subheader("ExpGrad — overall metrics")
        m = s.get("eval_metrics", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
        c2.metric("Precision", f"{m.get('precision', 0):.3f}")
        c3.metric("Recall", f"{m.get('recall', 0):.3f}")
        st.markdown("**Fairness gaps**")
        gaps = compute_gaps(s.get("group_metrics", {}))
        if gaps:
            gap_df = pd.Series(gaps).to_frame("value")
            gap_df["flag"] = gap_df["value"].apply(color_for_gap)
            st.table(gap_df.style.format("{:.4f}"))
    with tab_groups:
        st.subheader("Group metrics (expgrad)")
        st.dataframe(group_frame(s.get("group_metrics", {})).style.format("{:.4f}"))
    with tab_shap:
        st.subheader("ExpGrad SHAP")
        sp = s.get("shap_path")
        if sp and Path(sp).exists():
            st.image(sp, use_column_width=True)
        model = s.get("model")
        if isinstance(model, dict) and "clf" in model:
            clf = model["clf"]
            if clf.__class__.__name__ == "ExponentiatedGradient":
                if hasattr(clf, "weights_") and hasattr(clf, "predictors_"):
                    try:
                        w = np.array(clf.weights_)
                        idx = int(np.argmax(w))
                        st.info(f"ExpGrad: SHAP shown for top-weighted predictor #{idx} (weight={float(w[idx]):.3f})")
                    except Exception:
                        pass

# Run-all and show polished comparison
if run_all:
    with st.spinner("Running baseline..."):
        baseline = run_pipeline(use_mitigation=False)
    with st.spinner("Running reweight..."):
        reweight = run_pipeline(use_mitigation=True, mitigation_method="reweight")
    with st.spinner("Running expgrad..."):
        expgrad = run_pipeline(use_mitigation=True, mitigation_method="expgrad")
    st.success("All runs finished — comparison below")

    # metrics tab
    with tab_metrics:
        st.subheader("Overall metrics (baseline | reweight | expgrad)")
        df_overall = pd.DataFrame({
            "baseline": metrics_series(baseline.get("eval_metrics", {})),
            "reweight": metrics_series(reweight.get("eval_metrics", {})),
            "expgrad": metrics_series(expgrad.get("eval_metrics", {}))
        })
        st.table(df_overall.style.format("{:.3f}"))

        # fairness gaps
        gaps_b = compute_gaps(baseline.get("group_metrics", {}))
        gaps_r = compute_gaps(reweight.get("group_metrics", {}))
        gaps_e = compute_gaps(expgrad.get("group_metrics", {}))
        df_gaps = pd.DataFrame({
            "baseline": pd.Series(gaps_b),
            "reweight": pd.Series(gaps_r),
            "expgrad": pd.Series(gaps_e)
        })
        st.markdown("**Fairness gaps (smaller = fairer)**")
        def gap_mark(x): return f"{color_for_gap(x)} {x:.3f}"
        if not df_gaps.empty:
            display = df_gaps.fillna(0).applymap(lambda x: f"{x:.4f}")
            st.table(display)

    # groups tab
    with tab_groups:
        gb = group_frame(baseline.get("group_metrics", {})).add_suffix("_baseline")
        gr = group_frame(reweight.get("group_metrics", {})).add_suffix("_reweight")
        ge = group_frame(expgrad.get("group_metrics", {})).add_suffix("_expgrad")
        group_combined = pd.concat([gb, gr, ge], axis=1)
        st.subheader("Group metrics comparison")
        st.dataframe(group_combined.fillna("n/a").style.format("{:.4f}"))

    # shap tab (side-by-side)
    with tab_shap:
        st.subheader("SHAP comparison (baseline vs expgrad)")
        colA, colB = st.columns(2)
        with colA:
            st.write("Baseline SHAP")
            shp = baseline.get("shap_path")
            if shp and Path(shp).exists():
                st.image(shp, use_column_width=True)
        with colB:
            st.write("ExpGrad SHAP")
            shp2 = expgrad.get("shap_path")
            if shp2 and Path(shp2).exists():
                st.image(shp2, use_column_width=True)

    # download bundle
    zip_bytes = export_zip(df_overall, group_combined, df_gaps)
    st.download_button("⬇️ Download comparison (ZIP)", data=zip_bytes, file_name="fairness_comparison.zip", mime="application/zip")

# About tab content
with tab_about:
    st.markdown("### About this demo")
    st.markdown(
        """
        **FairLens** is a local demo that trains a classifier, examines per-group metrics,
        applies two simple mitigation strategies (reweighting and ExponentiatedGradient),
        and produces SHAP explanations to help interpret model behavior.
        """
    )
    st.markdown("**Tips:** Use `Use cache only` for deterministic offline demos. Run `Run all` to produce the side-by-side comparison used in applications.")
