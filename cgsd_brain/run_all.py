"""cgsd_brain.run_all — end-to-end entry point for the MVP smoke test.

Usage
-----
    # MVP smoke test (synthetic icosphere, 1 hemisphere, 1 atlas)
    python -m cgsd_brain.run_all --smoke

    # Phase A: real subject (fsaverage5 template) + real Schaefer 200 atlas
    python -m cgsd_brain.run_all --subject-id fsaverage5 --atlas schaefer-200

    # Phase B: real IXI subject (4-8h recon-all first)
    python -m cgsd_brain.run_all --subject-id IXI002 --atlas schaefer-200

    # Full run (Phase 2: 30 subjects × 2 hemi × 4 atlases × 3 seeds)
    python -m cgsd_brain.run_all --subject-id fsaverage5 --both-hemi

Pipeline
--------
    load_subject → build sulcal affinity → train sheaf → G3 cluster → evaluate

Atlas selection: tries real atlas first (e.g., Schaefer 200 .annot), falls
back to synthetic K-Means atlas if real files are missing (for MVP / CI).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

from cgsd_brain.data import load_subject
from cgsd_brain.sulcal import build_sulcal_affinity
from cgsd_brain.curvature import compute_forman_ricci
from cgsd_brain.train import train_cgsd_brain
from cgsd_brain.cluster import cluster_curvature_spectral
from cgsd_brain.atlas_eval import (
    load_synthetic_atlas,
    load_real_atlas,
    evaluate_nmi,
    evaluate_ari,
    evaluate_dice,
    ATLAS_K,
)


def _load_atlas(atlas_name: str, n_vertices: int) -> tuple[np.ndarray, str]:
    """Try real atlas first; fall back to synthetic if files missing.

    Returns
    -------
    labels : np.ndarray
    source : 'real' or 'synthetic'
    """
    try:
        labels = load_real_atlas(atlas_name, n_vertices=n_vertices)
        return labels, "real"
    except (FileNotFoundError, ValueError) as e:
        # FileNotFoundError: atlas files not downloaded (or wrong name)
        # ValueError: atlas name not in real atlas whitelist (synthetic-only atlas)
        try:
            labels = load_synthetic_atlas(atlas_name, n_vertices=n_vertices)
            print(f"  [atlas] {atlas_name} real unavailable → synthetic fallback")
            return labels, "synthetic"
        except ValueError:
            # Atlas name not in ATLAS_K either
            raise ValueError(
                f"Atlas {atlas_name!r} is neither a real atlas (no files) "
                f"nor in synthetic ATLAS_K. Original error: {e}"
            )


def run_one_hemisphere(
    sd,                     # SurfaceData
    gt_for_hemi: np.ndarray,
    K: int,
    config: dict,
    seed: int = 0,
) -> dict:
    """Run the full pipeline on one hemisphere, return metrics dict.

    Parameters
    ----------
    sd : SurfaceData
    gt_for_hemi : np.ndarray, shape (n_vertices,)
        Per-vertex ground-truth labels for THIS hemisphere (real or synthetic).
    K : int
        Number of parcels for this atlas.
    """
    n = sd.n_vertices

    # Step 1: Normalize features (3-channel per-vertex: sulc, curv, thickness).
    # This is the X that CSpec consumes on the surface mesh graph per §2.2.
    sulc_n = (sd.sulc - sd.sulc.mean()) / (sd.sulc.std() + 1e-8)
    curv_n = (sd.curv - sd.curv.mean()) / (sd.curv.std() + 1e-8)
    thk_n = (sd.thickness - sd.thickness.mean()) / (sd.thickness.std() + 1e-8)
    features = np.concatenate([
        sulc_n[:, None], curv_n[:, None], thk_n[:, None]
    ], axis=1).astype(np.float32)

    # Step 2: Build mesh adjacency
    adj = sd.build_mesh_adjacency()

    # Step 3: Build sulcal-depth-weighted affinity
    beta = config.get("sulcal_beta", 1.0)
    sulcal_adj = build_sulcal_affinity(sd.coords, sd.sulc, k=20, beta=beta)

    # Combine: A = max(mesh_adj, sulcal_affinity)
    combined_adj = adj.maximum(sulcal_adj)

    # Step 4: Get edge_index in COO format
    combined_coo = combined_adj.tocoo()
    edge_index = np.stack([combined_coo.row, combined_coo.col]).astype(np.int64)

    # Step 5: Compute Forman-Ricci curvature on the combined graph
    curvature = compute_forman_ricci(edge_index, num_nodes=n)

    # Step 6 (CSpec path): pass the 3-channel features X directly to
    # cluster_curvature_spectral — this is the §2.2 / §3.5 algorithm
    # described in the paper. No SheafDiffusion encoder is used; CSpec
    # is label-free by design.
    use_cspec_only = config.get("cspec_only", True)
    if use_cspec_only:
        Z = features
    else:
        # Legacy cgsd_brain MVP pipeline (SheafDiffusion + G3). Kept here
        # for completeness and for the Phase-A/B/C/D historical runs, but
        # NOT the algorithm reported in the JNM submission.
        Z = train_cgsd_brain(
            features=np.concatenate(
                [sd.coords, features], axis=1
            ).astype(np.float32),
            edge_index=edge_index,
            adj=combined_adj,
            hidden_dim=config.get("hidden_dim", 64),
            n_heads=config.get("n_heads", 2),
            n_epochs=config.get("struct_epochs", 30),
            lr=config.get("lr", 0.005),
            w_mod=config.get("w_mod", 1.0),
            w_col=config.get("w_col", 5.0),
            w_rec=config.get("w_rec", 1.0),
            w_sulcal=config.get("w_sulcal", 1.0),
            w_sym=0.0,
            seed=seed,
        )

    # Step 7: Curvature-aware spectral clustering (CSpec / G3)
    pred = cluster_curvature_spectral(
        Z, n_classes=K, curvature=curvature, edge_index=edge_index,
        alpha=config.get("alpha", 1.0),
        n_neighbors=config.get("k", 10),
        seed=seed,
    )

    # Step 8: Evaluate against atlas
    nmi = evaluate_nmi(pred, gt_for_hemi)
    ari = evaluate_ari(pred, gt_for_hemi)
    dice = evaluate_dice(pred, gt_for_hemi, n_classes=K)

    return {
        "nmi": nmi,
        "ari": ari,
        "dice_mean": float(dice.mean()),
        "dice_min": float(dice.min()),
        "K": K,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke test (synthetic subj, 1 hemi, 1 atlas)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--atlas", default="schaefer-200")
    parser.add_argument("--subject-id", default="synthetic_smoke",
                        help="synthetic_smoke | fsaverage5 (Phase A) | IXI### (Phase B)")
    parser.add_argument("--hemi", default="lh", choices=["lh", "rh"],
                        help="Ignored if --both-hemi is set")
    parser.add_argument("--both-hemi", action="store_true",
                        help="Run both hemispheres (Phase A template; Phase B)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())

    print(f"=== CGSD-Brain MVP smoke test ===")
    print(f"  subject: {args.subject_id}")
    print(f"  atlas:   {args.atlas}")
    print(f"  seed:    {args.seed}")

    t0 = time.time()
    subj = load_subject(args.subject_id)
    n_vertices = subj.lh.n_vertices
    print(f"  loaded surface: {n_vertices} vertices/hemi (subject_id={subj.subject_id})")

    # Load atlas once (try real, fallback to synthetic)
    atlas_labels_full, atlas_source = _load_atlas(args.atlas, n_vertices)
    K = ATLAS_K[args.atlas]
    # Per-hemisphere slice (real atlases: lh first 10242, rh next 10242;
    # synthetic: shape (n_vertices,) so use directly).
    if atlas_source == "real":
        assert atlas_labels_full.shape == (2 * n_vertices,), (
            f"real atlas shape {atlas_labels_full.shape} != (2 * {n_vertices},)"
        )
    print(f"  atlas loaded ({atlas_source}, K={K})")

    hemis = ["lh", "rh"] if args.both_hemi else [args.hemi]
    results_all = {}
    for hemi in hemis:
        sd = getattr(subj, hemi)
        if atlas_source == "real":
            if hemi == "lh":
                gt = atlas_labels_full[:n_vertices]
            else:
                gt = atlas_labels_full[n_vertices:]
        else:
            gt = atlas_labels_full  # synthetic is single-vector per-hemi
        m = run_one_hemisphere(sd, gt, K, config, seed=args.seed)
        results_all[hemi] = m
        print(f"\n  --- {hemi} ---")
        print(f"    NMI:  {m['nmi']:.4f}")
        print(f"    ARI:  {m['ari']:.4f}")
        print(f"    Dice: {m['dice_mean']:.4f} (mean) | {m['dice_min']:.4f} (min)")

    elapsed = time.time() - t0

    print(f"\n=== Results summary ===")
    print(f"  atlas:       {args.atlas} ({atlas_source})")
    print(f"  subject_id:  {subj.subject_id}")
    print(f"  K:           {K}")
    if len(hemis) == 1:
        m = results_all[hemis[0]]
        print(f"  NMI:  {m['nmi']:.4f}")
        print(f"  ARI:  {m['ari']:.4f}")
        print(f"  Dice: {m['dice_mean']:.4f} (mean) | {m['dice_min']:.4f} (min)")
    else:
        nmi_mean = np.mean([results_all[h]["nmi"] for h in hemis])
        ari_mean = np.mean([results_all[h]["ari"] for h in hemis])
        print(f"  NMI mean (lh+rh): {nmi_mean:.4f}")
        print(f"  ARI mean (lh+rh): {ari_mean:.4f}")
    print(f"  Time:        {elapsed:.1f} s")

    if args.smoke:
        # Smoke test only requires non-NaN finite metrics
        for h in hemis:
            assert np.isfinite(results_all[h]["nmi"]), f"{h} NMI not finite"
        print("\nSmoke test PASSED ✓")


if __name__ == "__main__":
    main()
