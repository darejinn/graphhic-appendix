"""Aggregate 40kb data to 1Mb resolution for Phase 4 GAE PoC."""

import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import json


def aggregate_to_1mb(project_root: str, cancer_type: str) -> dict:
    """Aggregate 40kb bins → 1Mb bins for one cancer type.
    
    Returns dict with all data needed for GAE training.
    """
    root = Path(project_root)
    data_dir = root / "data" / "processed" / cancer_type
    
    # --- 1. Bin mapping: 40kb → 1Mb ---
    bin_info = pd.read_csv(data_dir / "bin_info.csv")
    bin_info["mb_start"] = (bin_info["start"] // 1_000_000) * 1_000_000
    bin_info["mb_idx"] = bin_info.groupby(["chrom", "mb_start"]).ngroup()
    
    n_mb = bin_info["mb_idx"].nunique()
    mb_info = bin_info.groupby("mb_idx").agg(
        chrom=("chrom", "first"),
        start=("mb_start", "first"),
    ).reset_index()
    
    # Sparse aggregation matrix: (n_mb, n_40kb)
    agg_rows = bin_info["mb_idx"].values
    agg_cols = np.arange(len(bin_info))
    agg_mat = sparse.csr_matrix(
        (np.ones(len(bin_info)), (agg_rows, agg_cols)),
        shape=(n_mb, len(bin_info)),
    )
    
    # --- 2. Sample mutation matrices ---
    sample_list = sorted([
        f.stem for f in data_dir.glob("DO*.npz")
    ])
    
    V_samples = {}
    for sid in sample_list:
        V_40kb = sparse.load_npz(data_dir / f"{sid}.npz").toarray()  # (75918, 96)
        V_1mb = agg_mat @ V_40kb  # (n_mb, 96)
        V_samples[sid] = V_1mb.astype(np.float32)
    
    # --- 3. Graphs at 1Mb ---
    # 3a. 1D adjacency
    chroms_mb = mb_info["chrom"].values
    rows_1d, cols_1d = [], []
    for i in range(n_mb - 1):
        if chroms_mb[i] == chroms_mb[i + 1]:
            rows_1d.extend([i, i + 1])
            cols_1d.extend([i + 1, i])
    A_1d = sparse.csr_matrix(
        (np.ones(len(rows_1d)), (rows_1d, cols_1d)),
        shape=(n_mb, n_mb),
    )
    
    # 3b. Hi-C adjacency aggregated to 1Mb
    A_hic_40kb = sparse.load_npz(data_dir / "graphs" / "A_1d_hic.npz")
    A_1d_40kb = sparse.load_npz(data_dir / "graphs" / "A_1d.npz")
    A_hic_only = A_hic_40kb - A_1d_40kb
    A_hic_only.eliminate_zeros()
    A_hic_only.data = np.abs(A_hic_only.data)
    
    # Aggregate: A_1mb = agg_mat @ A_hic_only @ agg_mat.T
    A_hic_1mb = agg_mat @ A_hic_only @ agg_mat.T
    A_hic_1mb = (A_hic_1mb > 0).astype(np.float32)  # binarize
    A_hic_1mb.setdiag(0)
    A_hic_1mb.eliminate_zeros()
    
    # Combined: 1D + Hi-C
    A_combined = ((A_1d + A_hic_1mb) > 0).astype(np.float32)
    A_combined.setdiag(0)
    A_combined.eliminate_zeros()
    
    # --- 4. TAD assignment ---
    TAD_TISSUE = {
        "Skin-Melanoma": "IMR90",
        "Liver-HCC": "LI",
        "Eso-AdenoCa": "LG",
        "Panc-AdenoCA": "PA",
    }
    tissue = TAD_TISSUE.get(cancer_type, "IMR90")
    tad_file = root / "data" / "raw" / "hic" / "primary_cohort_TAD_boundaries" / f"{tissue}.IS.All_boundaries.bed"
    
    tad_assignment = np.full(n_mb, -1, dtype=np.int32)
    if tad_file.exists():
        tad_df = pd.read_csv(tad_file, sep="\t", header=None, names=["chrom", "start", "end"])
        tad_id = 0
        for _, row in tad_df.iterrows():
            mask = (
                (mb_info["chrom"] == row["chrom"]) &
                (mb_info["start"] >= row["start"]) &
                (mb_info["start"] < row["end"])
            )
            if mask.any():
                tad_assignment[mask.values] = tad_id
                tad_id += 1
    
    # --- 5. Epigenomic features at 1Mb ---
    epi = pd.read_csv(data_dir / "epigenomic_features.csv")
    repli = epi["repli_timing"].values
    comp = epi["compartment_score"].values
    repli_1mb = agg_mat @ repli / np.maximum(agg_mat.sum(axis=1).A1, 1)
    comp_1mb = agg_mat @ comp / np.maximum(agg_mat.sum(axis=1).A1, 1)
    
    # --- 6. GT exposures (SigProfiler) ---
    gt_file = root / "experiments" / "phase1" / "sigprofiler" / cancer_type / "assignment" / "gt_exposure.csv"
    gt_exposure = None
    if gt_file.exists():
        gt_exposure = pd.read_csv(gt_file, index_col=0)
    
    print(f"[{cancer_type}] 1Mb bins: {n_mb}, samples: {len(sample_list)}")
    print(f"  A_1d edges: {A_1d.nnz}, A_hic edges: {A_hic_1mb.nnz}, A_combined edges: {A_combined.nnz}")
    print(f"  TAD assigned: {(tad_assignment >= 0).sum()}/{n_mb}")
    
    return {
        "cancer_type": cancer_type,
        "n_bins": n_mb,
        "mb_info": mb_info,
        "V_samples": V_samples,
        "sample_list": sample_list,
        "A_1d": A_1d,
        "A_hic": A_hic_1mb,
        "A_combined": A_combined,
        "tad_assignment": tad_assignment,
        "repli_timing": repli_1mb.astype(np.float32),
        "compartment": comp_1mb.astype(np.float32),
        "gt_exposure": gt_exposure,
    }


def to_edge_index(A: sparse.csr_matrix) -> np.ndarray:
    """Convert sparse adjacency to PyG edge_index (2, E)."""
    coo = A.tocoo()
    return np.stack([coo.row, coo.col], axis=0).astype(np.int64)


def save_1mb_data(data: dict, out_dir: str):
    """Save aggregated 1Mb data to disk."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    data["mb_info"].to_csv(out / "mb_info.csv", index=False)
    sparse.save_npz(out / "A_1d.npz", data["A_1d"].tocsr())
    sparse.save_npz(out / "A_hic.npz", data["A_hic"].tocsr())
    sparse.save_npz(out / "A_combined.npz", data["A_combined"].tocsr())
    np.save(out / "tad_assignment.npy", data["tad_assignment"])
    np.save(out / "repli_timing.npy", data["repli_timing"])
    np.save(out / "compartment.npy", data["compartment"])
    
    # Save V matrices
    v_dir = out / "samples"
    v_dir.mkdir(exist_ok=True)
    for sid, V in data["V_samples"].items():
        np.save(v_dir / f"{sid}.npy", V)
    
    # Save sample list
    with open(out / "sample_list.txt", "w") as f:
        f.write("\n".join(data["sample_list"]))
    
    # Save GT
    if data["gt_exposure"] is not None:
        data["gt_exposure"].to_csv(out / "gt_exposure.csv")
    
    print(f"Saved to {out}")


if __name__ == "__main__":
    import sys
    project_root = "/home/darejin/GraphHiC"
    cancer_types = ["Skin-Melanoma", "Liver-HCC", "Eso-AdenoCa", "Panc-AdenoCA"]
    
    for ct in cancer_types:
        data = aggregate_to_1mb(project_root, ct)
        save_1mb_data(data, f"{project_root}/data/processed/{ct}/1mb")
