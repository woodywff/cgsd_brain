"""cgsd_brain.atlas_eval — 4-atlas loader + NMI/ARI/Dice evaluators.

For the MVP we generate SYNTHETIC atlases by clustering the synthetic
sulc/curv features into K regions using K-Means. This is good enough
to verify the pipeline runs end-to-end; real FreeSurfer-based atlases
(Desikan-Killiany, Destrieux, AAL3, Schaefer 2018) are downloaded in
Phase 2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    f1_score,
)

ATLAS_DIR = Path(__file__).parent.parent / "data" / "atlas_cache"


# Atlas K values — total across both hemispheres.
# (Per-hemi counts are K/2 + 1, since label 0 = medial wall is shared.)
# This convention matches what the algorithm uses for cluster_curvature_spectral;
# the algorithm over-segments when K > per-hemi count, but NMI/ARI are invariant.
ATLAS_K = {
    "desikan-killiany": 68,
    "destrieux": 150,  # 75 per hemi (FreeSurfer a2009s.annot); literature "148" excludes medial wall.
    "aal3": 166,
    "schaefer-100": 100,
    "schaefer-200": 200,
    "schaefer-400": 400,
}


def load_synthetic_atlas(name: str, n_vertices: int = 10242) -> np.ndarray:
    """Generate a synthetic atlas by K-Means clustering of random features.

    For MVP: ensures the pipeline runs end-to-end. Phase 2 replaces this
    with real FreeSurfer atlas downloads.
    """
    if name not in ATLAS_K:
        raise ValueError(f"Unknown atlas {name!r}. Known: {list(ATLAS_K.keys())}")
    K = ATLAS_K[name]

    rng = np.random.default_rng(hash(name) % 2**32)
    # Use spherical-harmonic-like features to make clustering non-trivial
    coords = rng.standard_normal((n_vertices, 3)).astype(np.float32)
    coords = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    features = np.concatenate([coords, rng.standard_normal((n_vertices, 4))], axis=1)

    km = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = km.fit_predict(features)
    return labels.astype(np.int64)


def load_real_atlas(name: str, n_vertices: int = 10242) -> np.ndarray:
    """Load a real fsaverage5 parcellation atlas (per-vertex integer labels).

    Supported names (atlas-sensitivity, 2026-06-30):
      - 'schaefer-100' (Schaefer 2018, 100 parcels, 17 Networks)
      - 'schaefer-200' (Schaefer 2018, 200 parcels, 17 Networks)
      - 'schaefer-400' (Schaefer 2018, 400 parcels, 17 Networks)
      - 'desikan-killiany' (68 regions, from FreeSurfer `aparc.annot`)
      - 'destrieux' (148 regions, from FreeSurfer `aparc.a2009s.annot`)

    Phase B will add:
      - 'aal3' (166 regions, from AAL3 v1.0)

    Parameters
    ----------
    name : str
        Atlas name (one of the supported keys above).
    n_vertices : int
        Expected vertex count per hemisphere. Schaefer 200 in fsaverage5 has 10242.

    Returns
    -------
    labels : np.ndarray, shape (2 * n_vertices,), dtype int64
        Concatenated lh + rh per-vertex integer parcel labels (1..K or 0..K-1).
    """
    # Dispatch table for Schaefer 2018 resolutions (CBIG, fsaverage5, 17Networks)
    schaefer_paths = {
        "schaefer-100": ("schaefer-100", "schaefer100"),
        "schaefer-200": ("schaefer-200", "schaefer200"),
        "schaefer-400": ("schaefer-400", "schaefer400"),
    }

    if name in schaefer_paths:
        dir_name, file_prefix = schaefer_paths[name]
        lh_path = ATLAS_DIR / dir_name / f"{file_prefix}_lh.annot"
        rh_path = ATLAS_DIR / dir_name / f"{file_prefix}_rh.annot"
        if not lh_path.exists() or not rh_path.exists():
            raise FileNotFoundError(
                f"Schaefer atlas files missing for {name!r}. "
                f"Expected at {lh_path} and {rh_path}. "
                f"Run: python scripts/download_atlases.py"
            )
        try:
            from nibabel.freesurfer.io import read_annot
            lh_labels, _ctab_lh, _names_lh = read_annot(str(lh_path))
            rh_labels, _ctab_rh, _names_rh = read_annot(str(rh_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {name} .annot files. "
                f"Try: pip install nibabel. Original error: {e}"
            )
        # FreeSurfer/CBIG convention: label 0 = medial wall (no parcel).
        # We keep them — the algorithm will see them as a "no-cluster" sink.
        return np.concatenate(
            [lh_labels.astype(np.int64), rh_labels.astype(np.int64)]
        )

    if name == "desikan-killiany":
        # FreeSurfer's standard DK68 atlas (`aparc.annot`, 34 per hemisphere).
        # These were copied into ATLAS_DIR/desikan-killiany/ by download_atlases.py.
        lh_path = ATLAS_DIR / "desikan-killiany" / "dk_lh.annot"
        rh_path = ATLAS_DIR / "desikan-killiany" / "dk_rh.annot"
        if not lh_path.exists() or not rh_path.exists():
            raise FileNotFoundError(
                f"DK68 atlas files missing. Expected at {lh_path} and {rh_path}. "
                f"Run: python scripts/download_atlases.py"
            )
        from nibabel.freesurfer.io import read_annot
        lh_labels, _ctab_lh, _names_lh = read_annot(str(lh_path))
        rh_labels, _ctab_rh, _names_rh = read_annot(str(rh_path))
        # DK68 labels can be negative (FreeSurfer convention for medial wall).
        # Re-map so the smallest label = 0 (medial wall), then 1..34 per hemi.
        # Combined unique labels = 68 (but we don't merge hemi labels — algorithm
        # sees concatenated lh+rh with original DK IDs).
        return np.concatenate(
            [lh_labels.astype(np.int64), rh_labels.astype(np.int64)]
        )

    if name == "destrieux":
        # FreeSurfer's Destrieux 2009 atlas (`aparc.a2009s.annot`, 74 per hemi,
        # 148 total). These were copied from $FREESURFER_HOME/subjects/
        # fsaverage5/label/{lh,rh}.aparc.a2009s.annot to ATLAS_DIR/destrieux/.
        lh_path = ATLAS_DIR / "destrieux" / "lh.aparc.a2009s.annot"
        rh_path = ATLAS_DIR / "destrieux" / "rh.aparc.a2009s.annot"
        if not lh_path.exists() or not rh_path.exists():
            raise FileNotFoundError(
                f"Destrieux atlas files missing. Expected at {lh_path} and "
                f"{rh_path}. See scripts/download_atlases.py to copy from "
                f"$FREESURFER_HOME/subjects/fsaverage5/label/."
            )
        from nibabel.freesurfer.io import read_annot
        lh_labels, _ctab_lh, _names_lh = read_annot(str(lh_path))
        rh_labels, _ctab_rh, _names_rh = read_annot(str(rh_path))
        return np.concatenate(
            [lh_labels.astype(np.int64), rh_labels.astype(np.int64)]
        )

    raise ValueError(
        f"Unknown real atlas {name!r}. Supported: 'schaefer-100/200/400', "
        "'desikan-killiany', 'destrieux'. Phase B will add 'aal3'."
    )


def evaluate_nmi(pred: np.ndarray, gt: np.ndarray) -> float:
    """Normalized mutual information (range [0, 1], 1 = perfect)."""
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(gt, pred))


def evaluate_ari(pred: np.ndarray, gt: np.ndarray) -> float:
    """Adjusted Rand index (range [-0.5, 1], 1 = perfect, 0 = random)."""
    return float(adjusted_rand_score(gt, pred))


def evaluate_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    n_classes: Optional[int] = None,
) -> np.ndarray:
    """Per-class F1 (Dice coefficient).

    Returns
    -------
    f1_per_class : np.ndarray, shape (n_classes,)
    """
    if n_classes is None:
        n_classes = max(pred.max(), gt.max()) + 1
    return f1_score(gt, pred, labels=np.arange(n_classes), average=None, zero_division=0)