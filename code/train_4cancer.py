"""Phase 4 v2: Train 8 cancer types with TAD reg=0 (no TAD regularization).

Saves all model weights, Z, and metadata to phase4_v2_no_tad/{cancer_type}/
"""
import sys
sys.path.insert(0, "/home/darejin/GraphHiC")

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from scipy.optimize import nnls
import json, time

from src.data.prepare_1mb import aggregate_to_1mb, to_edge_index
from src.models.gat_autoencoder import GraphSignatureAE
from src.utils.cosmic import load_cosmic_signatures, select_signatures_for_cancer_type, get_signature_matrix

PROJECT = "/home/darejin/GraphHiC"
OUT_BASE = Path(PROJECT) / "experiments" / "phase4_v2_no_tad"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HPARAMS = {
    "hidden": 64, "heads": 4, "dropout": 0.1, "lr": 1e-3,
    "epochs": 300, "lambda_tad": 0.0, "lambda_l2": 1e-4, "patience": 30,
}

# Cancer-specific signatures (PCAWG consensus)
CANCER_SIGS = {
    "Skin-Melanoma": ["SBS1", "SBS5", "SBS7a", "SBS7b", "SBS7c", "SBS7d", "SBS11", "SBS17a", "SBS17b", "SBS38"],
    "Liver-HCC": ["SBS1", "SBS4", "SBS5", "SBS6", "SBS12", "SBS16", "SBS17a", "SBS17b"],
    "Eso-AdenoCa": ["SBS1", "SBS2", "SBS3", "SBS5", "SBS13", "SBS17a", "SBS17b", "SBS18"],
    "Panc-AdenoCA": ["SBS1", "SBS2", "SBS3", "SBS5", "SBS13", "SBS17a", "SBS17b", "SBS18"],
    "Breast-AdenoCa": ["SBS1", "SBS2", "SBS3", "SBS5", "SBS8", "SBS13", "SBS17a", "SBS17b", "SBS18"],
    "Ovary-AdenoCA": ["SBS1", "SBS3", "SBS5", "SBS8", "SBS17a", "SBS17b", "SBS18"],
    "Prost-AdenoCA": ["SBS1", "SBS5", "SBS8", "SBS17a", "SBS17b", "SBS18"],
    "Stomach-AdenoCA": ["SBS1", "SBS2", "SBS5", "SBS13", "SBS17a", "SBS17b", "SBS18"],
}


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def train_and_save(name, use_graph, ei, samples, H_tensor, z_gt, hparams, device, out_dir):
    model = GraphSignatureAE(
        H_cosmic=H_tensor, hidden=hparams["hidden"],
        heads=hparams["heads"], dropout=hparams["dropout"],
        use_graph=use_graph,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=hparams["lr"])
    
    st = {s: torch.tensor(V, dtype=torch.float32, device=device) for s, V in samples.items()}
    sids = list(st.keys())
    e = ei.to(device) if ei is not None else None
    
    best, pc, bs = float("inf"), 0, None
    t0 = time.time()
    
    for ep in range(hparams["epochs"]):
        model.train()
        el = 0
        np.random.shuffle(sids)
        for s in sids:
            opt.zero_grad()
            vh, z = model(st[s], e)
            # IMPORTANT: lambda_tad=0
            l, d = model.loss(st[s], vh, z, lambda_tad=0.0, lambda_l2=hparams["lambda_l2"])
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            el += d["total"]
        el /= len(sids)
        
        if (ep+1) % 50 == 0:
            print(f"    {name} ep{ep+1} loss={el:.4f}")
        
        if el < best - 1e-4:
            best = el; pc = 0
            bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pc += 1
            if pc >= hparams["patience"]:
                print(f"    {name} early stop ep{ep+1}")
                break
    
    elapsed = time.time() - t0
    if bs:
        model.load_state_dict({k: v.to(device) for k, v in bs.items()})
    
    # Eval + save
    model.eval()
    Z_all = {}
    cos_sims = []
    with torch.no_grad():
        for s in sids:
            _, z = model(st[s], e)
            Z_all[s] = z.cpu().numpy()
            zs = z.sum(dim=0).cpu().numpy()
            zt = z_gt[s]
            if zt.sum() > 0 and zs.sum() > 0:
                cos_sims.append(cosine_sim(zs, zt))
    
    # Save
    torch.save(bs, out_dir / f"{name}_model.pt")
    np.savez_compressed(out_dir / f"{name}_Z.npz", **Z_all)
    
    mc = np.mean(cos_sims)
    sc = np.std(cos_sims)
    print(f"  {name}: cos={mc:.4f}±{sc:.4f}  time={elapsed:.0f}s")
    
    return {
        "cos_mean": float(mc), "cos_std": float(sc),
        "median": float(np.median(cos_sims)),
        "time_sec": elapsed, "best_loss": float(best),
    }


def run_cancer(ct):
    print(f"\n{'#'*60}\n# {ct}\n{'#'*60}")
    
    data = aggregate_to_1mb(PROJECT, ct)
    n_samples = len(data["V_samples"])
    
    cosmic = load_cosmic_signatures(f"{PROJECT}/data/reference/COSMIC_v3.4_SBS_GRCh37.txt")
    sig_candidates = CANCER_SIGS[ct]
    sig_names = [s for s in sig_candidates if s in cosmic.columns]
    print(f"Samples: {n_samples}, Signatures ({len(sig_names)}): {sig_names}")
    
    H = get_signature_matrix(cosmic, sig_names)
    H_tensor = torch.tensor(H, dtype=torch.float32)
    
    # NNLS GT
    z_gt = {sid: nnls(H.T, V.sum(axis=0))[0] for sid, V in data["V_samples"].items()}
    
    # Edge indices
    ei_1d = torch.tensor(to_edge_index(data["A_1d"]), dtype=torch.long)
    ei_combined = torch.tensor(to_edge_index(data["A_combined"]), dtype=torch.long)
    ei_dummy = torch.zeros((2, 0), dtype=torch.long)
    
    out_dir = OUT_BASE / ct
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "cancer_type": ct,
        "n_samples": n_samples,
        "n_bins": data["n_bins"],
        "n_signatures": len(sig_names),
        "signatures": sig_names,
        "hparams": HPARAMS,
        "models": {},
    }
    
    results["models"]["M_none"] = train_and_save(
        "M_none", False, ei_dummy, data["V_samples"], H_tensor, z_gt, HPARAMS, DEVICE, out_dir)
    results["models"]["M_1d"] = train_and_save(
        "M_1d", True, ei_1d, data["V_samples"], H_tensor, z_gt, HPARAMS, DEVICE, out_dir)
    results["models"]["M_hic"] = train_and_save(
        "M_hic", True, ei_combined, data["V_samples"], H_tensor, z_gt, HPARAMS, DEVICE, out_dir)
    
    # Summary
    d_1d = results["models"]["M_1d"]["cos_mean"] - results["models"]["M_none"]["cos_mean"]
    d_hic = results["models"]["M_hic"]["cos_mean"] - results["models"]["M_1d"]["cos_mean"]
    print(f"\n  {ct}: Δ(1d-none)={d_1d:+.4f}  Δ(hic-1d)={d_hic:+.4f}")
    
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_BASE}")
    print(f"Hyperparameters: {HPARAMS}")
    
    cancers = ["Skin-Melanoma", "Liver-HCC", "Eso-AdenoCa", "Panc-AdenoCA",
               "Breast-AdenoCa", "Ovary-AdenoCA", "Prost-AdenoCA", "Stomach-AdenoCA"]
    
    all_results = {}
    for ct in cancers:
        try:
            all_results[ct] = run_cancer(ct)
        except Exception as e:
            print(f"\n  ERROR in {ct}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"\n\n{'#'*70}\n# FINAL SUMMARY: 8 PCAWG cancers, λ_tad=0\n{'#'*70}")
    print(f"{'Cancer':<20} {'n':>5} {'M_none':>8} {'M_1d':>8} {'M_hic':>8} {'Δ(1d-none)':>12} {'Δ(hic-1d)':>12}")
    print("-" * 78)
    
    summary = {}
    for ct in cancers:
        if ct not in all_results: continue
        r = all_results[ct]
        m = r["models"]
        d1 = m["M_1d"]["cos_mean"] - m["M_none"]["cos_mean"]
        d2 = m["M_hic"]["cos_mean"] - m["M_1d"]["cos_mean"]
        print(f"{ct:<20} {r['n_samples']:>5} {m['M_none']['cos_mean']:>8.4f} "
              f"{m['M_1d']['cos_mean']:>8.4f} {m['M_hic']['cos_mean']:>8.4f} "
              f"{d1:>+12.4f} {d2:>+12.4f}")
        summary[ct] = {
            "n_samples": r["n_samples"],
            "M_none": m["M_none"]["cos_mean"],
            "M_1d": m["M_1d"]["cos_mean"],
            "M_hic": m["M_hic"]["cos_mean"],
            "delta_1d_none": d1,
            "delta_hic_1d": d2,
        }
    
    with open(OUT_BASE / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT_BASE}")


if __name__ == "__main__":
    main()
