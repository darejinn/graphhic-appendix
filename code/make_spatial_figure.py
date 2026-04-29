"""Per-sample spatial pattern evidence for the 4 proposal cancers.

Pulls totalZ vs Compartment PC1 (Spearman ρ) and SBS1 spatial CV
from the per-sample validation result JSON, restricted to 4 cancers.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

# This file is shipped under results/ so the appendix is self-contained.
PERSAMPLE = RES / "persample_results.json"

PROPOSAL_4 = ["Skin-Melanoma", "Liver-HCC", "Eso-AdenoCa", "Panc-AdenoCA"]
SHORT = {"Skin-Melanoma": "Skin",
         "Liver-HCC": "Liver",
         "Eso-AdenoCa": "Esophagus",
         "Panc-AdenoCA": "Pancreas"}

C_NONE, C_1D, C_HIC = "#9aa0a6", "#1f77b4", "#d62728"


def main():
    with open(PERSAMPLE) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), dpi=200)

    # --- Panel A: total-Z vs Compartment PC1 (per-sample mean ρ) ---
    ax = axes[0]
    x = np.arange(len(PROPOSAL_4))
    w = 0.27
    rho = {m: [] for m in ["M_none", "M_1d", "M_hic"]}
    for c in PROPOSAL_4:
        cmp = data[c]["correlations"]["totalZ_vs_comp"]
        for m in rho:
            rho[m].append(cmp[m]["rho_mean"])

    ax.bar(x - w, rho["M_none"], w, color=C_NONE, label="M_none")
    ax.bar(x,     rho["M_1d"],   w, color=C_1D,   label="M_1d")
    ax.bar(x + w, rho["M_hic"],  w, color=C_HIC,  label="M_hic")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in PROPOSAL_4], fontsize=9)
    ax.set_ylabel("Spearman ρ  (total Z  vs  Compartment PC1)")
    ax.set_title("(A) 3D structure preserved in latent Z\nopen chromatin → lower mutation load",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel B: SBS1 spatial CV (clock-like uniformity) ---
    ax = axes[1]
    cv = {m: [] for m in ["M_none", "M_1d", "M_hic"]}
    labels = []
    for c in PROPOSAL_4:
        sc = data[c]["sbs1_cv"]
        n_active = sc["M_hic"]["n_active"]
        if n_active == 0:
            # SBS1 collapsed in graph models — no measurable CV
            continue
        labels.append(SHORT[c])
        for m in cv:
            cv[m].append(sc[m]["cv_mean"])

    x2 = np.arange(len(labels))
    ax.bar(x2 - w, cv["M_none"], w, color=C_NONE, label="M_none")
    ax.bar(x2,     cv["M_1d"],   w, color=C_1D,   label="M_1d")
    ax.bar(x2 + w, cv["M_hic"],  w, color=C_HIC,  label="M_hic")
    ax.set_xticks(x2)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Per-sample SBS1 spatial CV (mean)")
    ax.set_title("(B) Clock-like SBS1 becomes more spatially uniform\n"
                 "(Skin omitted: SBS1 collapsed in M_1d/M_hic)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG / "fig_appendix_spatial_evidence.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
