"""
Create two demo screenshots (Pima + Adult) by running the pipeline for each dataset
and composing a polished PNG that shows the overall metrics table and the SHAP image.
Saves images to outputs/demo_pima.png and outputs/demo_adult.png.

Usage:
    source .venv/bin/activate
    python tools/create_demo_screenshots.py
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
from pmas.main import run_full_pipeline

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

def run_and_collect(dataset_name, use_cache_only=True, mitigation_method="reweight"):
    # baseline run
    state_baseline = run_full_pipeline(use_mitigation=False, dataset=dataset_name, use_cache_only=use_cache_only)
    # reweight run (optional; we only show baseline metrics + baseline SHAP here)
    state_reweight = run_full_pipeline(use_mitigation=True, mitigation_method="reweight", dataset=dataset_name, use_cache_only=use_cache_only)
    # expgrad run (optional)
    state_exp = run_full_pipeline(use_mitigation=True, mitigation_method="expgrad", dataset=dataset_name, use_cache_only=use_cache_only)

    return {
        "baseline": state_baseline,
        "reweight": state_reweight,
        "expgrad": state_exp
    }

def metrics_to_df(baseline_s, reweight_s, exp_s):
    b = baseline_s.get("eval_metrics", {}) or {}
    r = reweight_s.get("eval_metrics", {}) or {}
    e = exp_s.get("eval_metrics", {}) or {}
    df = pd.DataFrame({
        "baseline": pd.Series(b),
        "reweight": pd.Series(r),
        "expgrad": pd.Series(e)
    })
    return df

def compose_png(dataset_label, states, out_path: Path):
    # compose a figure with the overall metrics table on top and the SHAP image below (if exists)
    baseline = states["baseline"]
    reweight = states["reweight"]
    expgrad = states["expgrad"]

    df_overall = metrics_to_df(baseline, reweight, expgrad).fillna("")

    # load SHAP image (prefer baseline SHAP)
    shap_path = baseline.get("shap_path") or reweight.get("shap_path") or expgrad.get("shap_path")
    shap_img = None
    if shap_path and Path(shap_path).exists():
        shap_img = Image.open(shap_path).convert("RGBA")
    # layout sizes
    width = 1400
    table_h = 220
    shap_h = 520 if shap_img is not None else 200
    height = table_h + shap_h + 60

    canvas = Image.new("RGB", (width, height), (250, 252, 255))
    # draw table using matplotlib
    fig, ax = plt.subplots(figsize=(width/100, table_h/100), dpi=100)
    ax.axis('off')
    ax.set_title(f"{dataset_label} — Overall metrics (baseline | reweight | expgrad)", fontsize=16, pad=12)
    tbl = ax.table(cellText=df_overall.round(3).fillna("").values,
                   colLabels=df_overall.columns,
                   rowLabels=df_overall.index,
                   cellLoc='center',
                   colLoc='center',
                   loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.6)
    fig.canvas.draw()

    # paste the table into canvas
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    table_img = Image.fromarray(data)
    plt.close(fig)

    # place table at top
    canvas.paste(table_img, (20, 10))

    # place SHAP image below (centered)
    if shap_img is not None:
        # resize shap to fit width/height
        max_w = width - 160
        max_h = shap_h - 40
        ratio = min(max_w / shap_img.width, max_h / shap_img.height, 1.0)
        new_size = (int(shap_img.width * ratio), int(shap_img.height * ratio))
        shap_small = shap_img.resize(new_size, Image.LANCZOS)
        x = (width - new_size[0]) // 2
        y = 20 + table_h + 10
        canvas.paste(shap_small, (x, y), shap_small if shap_small.mode == 'RGBA' else None)
    else:
        # add placeholder text
        fig2, ax2 = plt.subplots(figsize=(width/100, shap_h/100), dpi=100)
        ax2.text(0.5, 0.5, "No SHAP image available", horizontalalignment='center', verticalalignment='center', fontsize=16, color='gray')
        ax2.axis('off')
        fig2.canvas.draw()
        data2 = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
        data2 = data2.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
        placeholder = Image.fromarray(data2)
        plt.close(fig2)
        canvas.paste(placeholder, (20, 20 + table_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)
    print(f"Wrote demo screenshot: {out_path}")

def main():
    datasets = [("Pima Diabetes", True), ("Adult Income", True)]
    for ds_label, cache_only in datasets:
        print(f"Running pipeline for {ds_label} (cache_only={cache_only}) ...")
        states = run_and_collect(ds_label, use_cache_only=cache_only)
        out_name = "demo_pima.png" if "Pima" in ds_label else "demo_adult.png"
        out_path = OUT_DIR / out_name
        compose_png(ds_label, states, out_path)

if __name__ == "__main__":
    main()
