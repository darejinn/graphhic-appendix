"""Generate appendix figures focused on the 4 proposal cancer types.

Inputs : ../results/summary_table.csv  (8 PCAWG cancers)
         ../results/{cancer}_results.json  (per-cancer detail)
Outputs: ../figures/fig_appendix_*.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

PROPOSAL_4 = ["Skin-Melanoma", "Liver-HCC", "Eso-AdenoCa", "Panc-AdenoCA"]
SHORT = {"Skin-Melanoma": "Skin\n(Melanoma)",
         "Liver-HCC": "Liver\n(HCC)",
         "Eso-AdenoCa": "Esophagus\n(AdenoCa)",
         "Panc-AdenoCA": "Pancreas\n(AdenoCA)"}

C_NONE, C_1D, C_HIC = "#9aa0a6", "#1f77b4", "#d62728"


def fig_cosine_4cancer():
    df = pd.read_csv(RES / "summary_table.csv")
    df = df.set_index("cancer_type").loc[PROPOSAL_4].reset_index()

    x = np.arange(len(df))
    w = 0.27

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.bar(x - w, df["M_none_mean"], w, yerr=df["M_none_std"],
           color=C_NONE, label="M_none (MLP, no graph)", capsize=3)
    ax.bar(x,     df["M_1d_mean"],   w, yerr=df["M_1d_std"],
           color=C_1D,   label="M_1d (GAT + 1D adjacency)", capsize=3)
    ax.bar(x + w, df["M_hic_mean"],  w, yerr=df["M_hic_std"],
           color=C_HIC,  label="M_hic (GAT + 1D + Hi-C)",   capsize=3)

    for i, r in df.iterrows():
        ax.text(i + w, r["M_hic_mean"] + 0.005, f"{r['M_hic_mean']:.3f}",
                ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{SHORT[c]}\n(n={int(n)})"
                        for c, n in zip(df["cancer_type"], df["n_samples"])],
                       fontsize=9)
    ax.set_ylim(0.85, 1.005)
    ax.set_ylabel("Cosine similarity (Z·H vs NNLS GT)")
    ax.set_title("GraphHiC — bin-level reconstruction quality (4 proposal cancers)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_appendix_cosine_4cancer.png", bbox_inches="tight")
    print(f"  -> {FIG / 'fig_appendix_cosine_4cancer.png'}")


def fig_delta_4cancer():
    df = pd.read_csv(RES / "summary_table.csv")
    df = df.set_index("cancer_type").loc[PROPOSAL_4].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), dpi=200, sharey=True)

    y = np.arange(len(df))
    for ax, col, color, title in [
        (axes[0], "delta_1d_none", C_1D,
         "Δ(M_1d − M_none)\nGAT + 1D adjacency vs MLP"),
        (axes[1], "delta_hic_1d", C_HIC,
         "Δ(M_hic − M_1d)\nadding Hi-C edges"),
    ]:
        bars = ax.barh(y, df[col], color=color, alpha=0.85)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[c].replace("\n", " ") for c in df["cancer_type"]],
                           fontsize=9)
        for b, v in zip(bars, df[col]):
            ax.text(v + (0.001 if v >= 0 else -0.001), b.get_y() + b.get_height()/2,
                    f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlabel("Δ cosine")

    fig.suptitle("Per-step contribution of graph & Hi-C", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_appendix_delta_4cancer.png", bbox_inches="tight")
    print(f"  -> {FIG / 'fig_appendix_delta_4cancer.png'}")


def fig_signature_grid():
    fig, ax = plt.subplots(figsize=(7.5, 2.8), dpi=200)
    sigs_all = []
    for c in PROPOSAL_4:
        with open(RES / f"{c}_results.json") as f:
            sigs_all.extend(json.load(f)["signatures"])
    sig_order = sorted(set(sigs_all),
                       key=lambda s: (int("".join(filter(str.isdigit, s)) or 0),
                                      s))

    M = np.zeros((len(PROPOSAL_4), len(sig_order)), dtype=int)
    for i, c in enumerate(PROPOSAL_4):
        with open(RES / f"{c}_results.json") as f:
            for s in json.load(f)["signatures"]:
                M[i, sig_order.index(s)] = 1

    ax.imshow(M, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(sig_order)))
    ax.set_xticklabels(sig_order, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(PROPOSAL_4)))
    ax.set_yticklabels([SHORT[c].replace("\n", " ") for c in PROPOSAL_4],
                       fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j]:
                ax.text(j, i, "●", ha="center", va="center",
                        color="white", fontsize=8)
    ax.set_title("Active COSMIC SBS signatures per cancer (decoder dimension k)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_appendix_signatures.png", bbox_inches="tight")
    print(f"  -> {FIG / 'fig_appendix_signatures.png'}")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_cosine_4cancer()
    fig_delta_4cancer()
    fig_signature_grid()
    print("done.")
