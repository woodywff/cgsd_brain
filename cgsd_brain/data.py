"""cgsd_brain.data — surface loaders for IXI + synthetic + fsaverage5 template.

Supported subjects:
- 'synthetic_smoke': bundled synthetic icosphere (MVP smoke test).
- 'fsaverage5':       FreeSurfer 8.2.0 template (population averages — Phase A).
- 'IXI###':           real subject under $SUBJECTS_DIR/ixi###/ (Phase B).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

# Default data location (relative to project root)
DATA_DIR = Path(__file__).parent.parent / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic_surface"

# FreeSurfer install paths to try (in priority order).
FS_SUBJECTS_CANDIDATES = [
    Path("/usr/local/freesurfer/8.2.0/subjects"),
    Path("/usr/local/freesurfer/subjects"),
]


def _resolve_fs_template_dir(name: str = "fsaverage5") -> Path:
    """Find FreeSurfer's fsaverage5 (or fsaverage) template directory."""
    for fs_root in FS_SUBJECTS_CANDIDATES:
        candidate = fs_root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"FreeSurfer template {name!r} not found. Tried: "
        + ", ".join(str(p / name) for p in FS_SUBJECTS_CANDIDATES)
    )


@dataclass
class SurfaceData:
    """Per-vertex data for one cortical hemisphere surface."""
    coords: np.ndarray       # (N, 3) fsaverage5 3D coords
    faces: np.ndarray        # (F, 3) triangle indices
    sulc: np.ndarray         # (N,) sulcal depth (FreeSurfer `sulc`)
    curv: np.ndarray         # (N,) mean curvature (FreeSurfer `curv`)
    thickness: np.ndarray    # (N,) cortical thickness in mm
    hemisphere: str          # "lh" or "rh"

    @property
    def n_vertices(self) -> int:
        return self.coords.shape[0]

    def build_mesh_adjacency(self) -> csr_matrix:
        """Build a symmetric mesh adjacency matrix (each edge = 2 directed)."""
        n = self.n_vertices
        rows = self.faces[:, 0]
        cols = self.faces[:, 1]
        # Triangle (a, b, c) gives 3 undirected edges: a-b, b-c, c-a
        edges = np.concatenate([
            np.stack([rows, cols], axis=1),
            np.stack([cols, self.faces[:, 2]], axis=1),
            np.stack([self.faces[:, 2], rows], axis=1),
        ])
        # Symmetric: (i, j) and (j, i)
        data = np.ones(edges.shape[0] * 2, dtype=np.float32)
        rows_full = np.concatenate([edges[:, 0], edges[:, 1]])
        cols_full = np.concatenate([edges[:, 1], edges[:, 0]])
        return csr_matrix((data, (rows_full, cols_full)), shape=(n, n))


@dataclass
class Subject:
    """One subject = left hemisphere + right hemisphere."""
    subject_id: str
    lh: SurfaceData
    rh: SurfaceData


class SyntheticSubject:
    """In-memory synthetic subject (no FastSurfer needed for MVP smoke)."""

    def __init__(self):
        lh_npz = np.load(SYNTHETIC_DIR / "synthetic_lh.npz")
        rh_npz = np.load(SYNTHETIC_DIR / "synthetic_rh.npz")
        self._subject_id = "synthetic_smoke"
        self.lh = SurfaceData(
            coords=lh_npz["coords"],
            faces=lh_npz["faces"],
            sulc=lh_npz["sulc"],
            curv=lh_npz["curv"],
            thickness=lh_npz["thickness"],
            hemisphere="lh",
        )
        self.rh = SurfaceData(
            coords=rh_npz["coords"],
            faces=rh_npz["faces"],
            sulc=rh_npz["sulc"],
            curv=rh_npz["curv"],
            thickness=rh_npz["thickness"],
            hemisphere="rh",
        )

    @property
    def subject_id(self) -> str:
        return self._subject_id


class FSSubject:
    """Real FreeSurfer subject (Phase A: fsaverage5 template; Phase B: real recon-all).

    Reads per-vertex sulc/curv/thickness from FreeSurfer subject's surf/ dir.
    Faces (topology) are loaded from the fsaverage5 template's lh.white/rh.white
    — these are FIXED across subjects (all use the same fsaverage5 sphere topology).
    """

    def __init__(self, subject_id: str, source_space: str = "fs5"):
        from cgsd_brain.fs_extract import extract_subject_fs5

        self._subject_id = subject_id
        # Map subject_id → recon_dir
        if subject_id.lower() in ("fsaverage5", "fsaverage"):
            recon_dir = _resolve_fs_template_dir(subject_id.lower())
        elif subject_id.upper().startswith("IXI"):
            # Use the original digits (e.g., "002") for the recon-all output
            # dir name. Don't strip leading zeros — recon_all_runner.sh
            # takes the second positional arg verbatim, so the recon dir
            # matches what the user passed in.
            num = subject_id.upper().replace("IXI", "")
            subjects_dir = Path(
                os.environ.get("SUBJECTS_DIR", str(Path.home() / "subjects"))
            )
            recon_dir = subjects_dir / f"ixi{num}"
            if not recon_dir.exists():
                raise FileNotFoundError(
                    f"No recon-all output at {recon_dir}. "
                    f"Run: bash scripts/recon_all_runner.sh <t1.nii.gz> ixi{num}"
                )
        else:
            raise ValueError(
                f"Unsupported FSSubject id {subject_id!r}. "
                "Use 'fsaverage5' (Phase A template) or 'IXI###' (Phase B real subject)."
            )

        lh_d, rh_d = extract_subject_fs5(Path(recon_dir), source_space=source_space)

        # Load fsaverage5 topology (faces). All fsaverage5 subjects share the same
        # triangle mesh, so we read it once from the template's lh.white.
        fs_tpl_dir = _resolve_fs_template_dir("fsaverage5")
        from nibabel.freesurfer.io import read_geometry
        _, faces_lh = read_geometry(str(fs_tpl_dir / "surf" / "lh.white"))
        _, faces_rh = read_geometry(str(fs_tpl_dir / "surf" / "rh.white"))

        self.lh = SurfaceData(
            coords=lh_d["coords"], faces=faces_lh,
            sulc=lh_d["sulc"], curv=lh_d["curv"], thickness=lh_d["thickness"],
            hemisphere="lh",
        )
        self.rh = SurfaceData(
            coords=rh_d["coords"], faces=faces_rh,
            sulc=rh_d["sulc"], curv=rh_d["curv"], thickness=rh_d["thickness"],
            hemisphere="rh",
        )

    @property
    def subject_id(self) -> str:
        return self._subject_id


def load_subject(subject_id: str) -> Subject:
    """Dispatch by subject_id.

    Supported subjects:
    - 'synthetic_smoke'  — built-in synthetic icosphere (MVP smoke test).
    - 'fsaverage5'       — FreeSurfer 8.2.0 template (Phase A, population averages).
    - 'IXI###'           — real subject under $SUBJECTS_DIR/ixi###/ (Phase B).
    """
    if subject_id == "synthetic_smoke":
        s = SyntheticSubject()
        return Subject(subject_id=s.subject_id, lh=s.lh, rh=s.rh)
    if subject_id.lower() in ("fsaverage5", "fsaverage"):
        s = FSSubject("fsaverage5")
        return Subject(subject_id=s.subject_id, lh=s.lh, rh=s.rh)
    if subject_id.upper().startswith("IXI"):
        s = FSSubject(subject_id)
        return Subject(subject_id=s.subject_id, lh=s.lh, rh=s.rh)
    raise NotImplementedError(
        f"subject_id={subject_id!r} not supported. "
        "Phase 2 supports 'synthetic_smoke', 'fsaverage5', and 'IXI###'."
    )