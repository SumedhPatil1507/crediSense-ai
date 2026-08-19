"""
Generate high-resolution README artifacts.
Run: python -m src.generate_readme_artifacts

Outputs:
  assets/shap_beeswarm.png      — SHAP global feature importance
  assets/model_evaluation.png   — ROC-AUC + Precision-Recall side-by-side
  assets/fairness_audit.png     — Disparate Impact Ratio subgroup analysis
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BRAND_BLUE  = "#1f77b4"
BRAND_RED   = "#d62728"
BRAND_GREEN = "#2ca02c"
BRAND_ORANGE= "#ff7f0e"


def _load_data():
    from src.data_loader import load_data
    from src.preprocessing import clean_data
    from src.feature_engineering import create_features
    from sklearn.model_selection import train_test_split

    print("Loading dataset…")
    df = load_data(str(BASE_DIR / "data" / "loan_cleaned.csv"))
    df = clean_data(df)
    df = create_features(df)
    X = df.drop(columns=["Risk_Flag"])
    y = df["Risk_Flag"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 1. SHAP BEESWARM
# ─────────────────────────────────────────────────────────────────────────────

def generate_shap_beeswarm(model, X_test):
    import shap

    print("Computing SHAP values (200 samples)…")
    pre = model.named_steps["preprocessor"]
    mod = model.named_steps["model"]

    sample = X_test.head(200)
    X_t = pre.transform(sample)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    feat_names = pre.get_feature_names_out().tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(mod)
        sv = explainer.shap_values(X_t)
    sv = sv[1] if isinstance(sv, list) else sv

    # Clean feature names for display
    clean_names = [
        n.replace("num__", "").replace("cat__", "").replace("_", " ").title()
        for n in feat_names
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_t, feature_names=clean_names, show=False,
                      plot_size=None, color_bar=True, max_display=15)
    ax = plt.gca()
    ax.set_xlabel("SHAP Value  (impact on default probability)", fontsize=11)
    ax.set_title("Global Feature Importance — SHAP Beeswarm\n"
                 "LightGBM Credit Risk Model  |  n=200 applicants", fontsize=13)
    fig.tight_layout()
    out = ASSETS_DIR / "shap_beeswarm.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL EVALUATION (ROC + PR side-by-side)
# ─────────────────────────────────────────────────────────────────────────────

def generate_model_evaluation(model, X_test, y_test):
    from sklearn.metrics import (roc_curve, roc_auc_score,
                                  precision_recall_curve, average_precision_score)

    print("Computing evaluation curves…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _     = roc_curve(y_test, y_prob)
    auc             = roc_auc_score(y_test, y_prob)
    prec, rec, _    = precision_recall_curve(y_test, y_prob)
    pr_auc          = average_precision_score(y_test, y_prob)
    baseline        = y_test.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ROC curve
    ax1.plot(fpr, tpr, color=BRAND_BLUE, lw=2.5,
             label=f"LightGBM  AUC = {auc:.4f}")
    ax1.plot([0, 1], [0, 1], color="grey", lw=1.2, linestyle="--", label="Random  AUC = 0.50")
    ax1.fill_between(fpr, tpr, alpha=0.08, color=BRAND_BLUE)
    ax1.set_xlabel("False Positive Rate (1 − Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity)")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1.02])

    # Annotation box
    gini = round(2 * auc - 1, 4)
    ax1.text(0.52, 0.18,
             f"AUC  = {auc:.4f}\nGini = {gini:.4f}",
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor=BRAND_BLUE, alpha=0.9))

    # PR curve
    ax2.step(rec, prec, color=BRAND_ORANGE, lw=2.5, where="post",
             label=f"LightGBM  PR-AUC = {pr_auc:.4f}")
    ax2.axhline(baseline, color="grey", lw=1.2, linestyle="--",
                label=f"Baseline (default rate = {baseline:.2%})")
    ax2.fill_between(rec, prec, step="post", alpha=0.08, color=BRAND_ORANGE)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="upper right")
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])

    ax2.text(0.04, 0.15,
             f"PR-AUC = {pr_auc:.4f}\nClass imbalance = {baseline:.1%}",
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor=BRAND_ORANGE, alpha=0.9))

    fig.suptitle("Model Evaluation  |  LightGBM on 20% Holdout Set  (n ≈ 50 400)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out = ASSETS_DIR / "model_evaluation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FAIRNESS AUDIT (DIR bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def generate_fairness_audit(model, X_test, y_test):
    from src.fairness import subgroup_metrics, disparate_impact_ratio

    print("Computing fairness metrics…")
    df_test = X_test.copy()
    df_test["Risk_Flag"] = y_test.values

    metrics_df  = subgroup_metrics(model, df_test)
    dir_df      = disparate_impact_ratio(metrics_df)

    if metrics_df.empty:
        print("  No subgroup data — skipping fairness plot.")
        return

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)

    # ── left: approval rate by subgroup ──────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0])
    groups   = metrics_df["group_col"].unique()
    n_groups = len(groups)
    palette  = [BRAND_BLUE, BRAND_GREEN, BRAND_ORANGE, "#9467bd"]

    bar_data = []
    for i, grp in enumerate(groups):
        sub = metrics_df[metrics_df["group_col"] == grp].sort_values("group_val")
        bar_data.append((grp, sub))

    y_pos = 0
    y_ticks, y_labels = [], []
    bar_height = 0.55
    for i, (grp, sub) in enumerate(bar_data):
        for _, row in sub.iterrows():
            color = palette[i % len(palette)]
            bar = ax_bar.barh(y_pos, row["approval_rate"], height=bar_height,
                               color=color, alpha=0.85, label=grp if y_pos == i else "")
            ax_bar.text(row["approval_rate"] + 0.005, y_pos,
                        f"{row['approval_rate']:.1%}", va="center", fontsize=9)
            y_ticks.append(y_pos)
            y_labels.append(f"{row['group_val']}\n({grp})")
            y_pos += 1
        y_pos += 0.4  # gap between groups

    ax_bar.axvline(0.80, color=BRAND_RED, lw=1.8, linestyle="--",
                   label="4/5ths Rule threshold (80%)")
    ax_bar.set_xlabel("Approval Rate")
    ax_bar.set_title("Approval Rate by Demographic Subgroup")
    ax_bar.set_yticks(y_ticks)
    ax_bar.set_yticklabels(y_labels, fontsize=9)
    ax_bar.set_xlim(0, 1.08)
    ax_bar.legend(loc="lower right", fontsize=9)

    # ── right: DIR table ─────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")

    if not dir_df.empty:
        rows   = []
        colors = []
        for _, row in dir_df.iterrows():
            ratio = row.get("disparate_impact_ratio")
            fair  = row.get("fair", "?")
            rows.append([
                row["group_col"],
                f"{ratio:.3f}" if ratio else "N/A",
                fair
            ])
            colors.append(["#ffffff", "#ffffff",
                            "#d4edda" if fair == "Yes" else "#f8d7da"])

        tbl = ax_tbl.table(
            cellText=rows,
            colLabels=["Group", "DIR", "Fair?"],
            cellColours=colors,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.1, 2.0)
        ax_tbl.set_title("Disparate Impact Ratio\n(≥ 0.80 = compliant)", fontsize=11)

    fig.suptitle("Fairness Audit  |  ECOA 4/5ths Rule  |  LightGBM Credit Risk Model",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = ASSETS_DIR / "fairness_audit.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CrediSense AI — README Artifact Generator")
    print("=" * 60)

    print("\nLoading model…")
    model = joblib.load(BASE_DIR / "models" / "model.pkl")

    X_test, y_test = _load_data()
    print(f"Test set: {len(X_test):,} rows")

    print("\n[1/3] SHAP Beeswarm")
    generate_shap_beeswarm(model, X_test)

    print("\n[2/3] Model Evaluation")
    generate_model_evaluation(model, X_test, y_test)

    print("\n[3/3] Fairness Audit")
    generate_fairness_audit(model, X_test, y_test)

    print("\nDone. Assets written to:", ASSETS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
