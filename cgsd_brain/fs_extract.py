"""cgsd_brain.fs_extract — FreeSurfer output → per-vertex numpy arrays.

Two input paths supported:

1. **fsaverage5 template** (Phase A, free):
   Path: /usr/local/freesurfer/8.2.0/subjects/fsaverage5/surf/{lh,rh}.{sulc,curv,thickness}
   Format: FreeSurfer binary morph data (.mgz). Read via
           nibabel.freesurfer.io.read_morph_data.
   Vertices: 10242 per hemisphere (already in fsaverage5 space).

2. **Real subject remapped to fsaverage5** (Phase B):
   Path: <subject_dir>/surf/{lh,rh}.fs5.{sulc,curv,thickness}
   Format: .mgh written by mri_surf2surf (Task A.6, native FreeSurfer 8.2.0).
   Vertices: 10242 per hemisphere (remapped from subject's native space).

The module auto-detects which format is present at <recon_dir>/surf/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def read_morph_per_vertex(path: Path) -> np.ndarray:
    """Read a FreeSurfer per-vertex morph file as a 1D float32 array.

    Two formats supported:
    - **FreeSurfer morph data** (no extension, e.g. lh.sulc on the fsaverage5
      template): parsed by nibabel.freesurfer.io.read_morph_data. Header-less
      float32 array, length = number of vertices.
    - **mri_surf2surf MGH v2** (gzip-compressed, 284-byte header, big-endian;
      e.g. lh.fs5.sulc on a remapped subject): nibabel 5.x rejects this format
      ("Unknown MGH format version"). We parse the header + data directly.

    Parameters
    ----------
    path : Path
        Path to morph data file (e.g., lh.sulc, lh.fs5.sulc).

    Returns
    -------
    arr : np.ndarray, shape (N,)
        Flat 1D array of per-vertex values (N = number of vertices on the
        mesh the file was written for).
    """
    import gzip
    import struct
    # Try MGH v2 (gzip + 284-byte BE header) first
    try:
        with gzip.open(str(path), 'rb') as f:
            raw = f.read()
        if len(raw) >= 284:
            hdr_ints = struct.unpack('>7i', raw[:28])
            version, width, height, depth, nframes, dtype_mgh, dof = hdr_ints
            # MGH v2 has version=513 (0x00000201). dtype_mgh==3 means FLOAT.
            if dtype_mgh == 3 and width > 0:
                arr = np.frombuffer(
                    raw[284:284 + width * 4], dtype='>f4'
                ).astype('<f4')
                if arr.shape[0] == width:
                    return arr
    except (OSError, EOFError, struct.error):
        pass

    # Fall back to FreeSurfer morph data (header-less float32 array)
    from nibabel.freesurfer.io import read_morph_data
    return read_morph_data(str(path)).astype(np.float32)


def _detect_format(recon_dir: Path) -> Dict[str, str]:
    """Detect which FreeSurfer format is in <recon_dir>/surf/.

    Returns dict mapping hemi → "fs5_template" or "fs5_remapped".
    Raises FileNotFoundError if neither format is present.
    """
    surf_dir = Path(recon_dir) / "surf"
    if not surf_dir.exists():
        raise FileNotFoundError(f"No surf/ dir under {recon_dir}")
    out: Dict[str, str] = {}
    for hemi in ["lh", "rh"]:
        template_file = surf_dir / f"{hemi}.sulc"      # fsaverage5 template uses .sulc
        remapped_file = surf_dir / f"{hemi}.fs5.sulc"  # mri_surf2surf output
        if template_file.exists() and not remapped_file.exists():
            out[hemi] = "fs5_template"
        elif remapped_file.exists():
            out[hemi] = "fs5_remapped"
        else:
            raise FileNotFoundError(
                f"No sulc/curv/thickness data found for {hemi} in {surf_dir}. "
                f"Expected either {hemi}.sulc (fsaverage5 template) or "
                f"{hemi}.fs5.sulc (remapped subject)."
            )
    return out


def _load_fsaverage5_coords(source_space: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load fsaverage{5,7} sphere vertex coordinates.

    Tries native FreeSurfer install first (preferred). Falls back to:
    - cached .npz at data/fsaverage5_sphere.npz, or
    - synthesized trimesh icosphere (last resort, slightly wrong topology).
    """
    from nibabel.freesurfer.io import read_geometry
    fs_home_candidates = [
        Path("/usr/local/freesurfer/8.2.0/subjects"),
        Path("/usr/local/freesurfer/subjects"),
    ]
    sphere_name = "fsaverage5" if source_space == "fs5" else "fsaverage"
    for fs_home in fs_home_candidates:
        sphere_lh = fs_home / sphere_name / "surf" / "lh.sphere"
        sphere_rh = fs_home / sphere_name / "surf" / "rh.sphere"
        if sphere_lh.exists() and sphere_rh.exists():
            coords_lh, _ = read_geometry(str(sphere_lh))
            coords_rh, _ = read_geometry(str(sphere_rh))
            return coords_lh.astype(np.float32), coords_rh.astype(np.float32)

    # Fallback: cached .npz from data/fsaverage5_sphere.npz
    coords_cache = (
        Path(__file__).parent.parent / "data" / "fsaverage5_sphere.npz"
    )
    if coords_cache.exists():
        npz = np.load(coords_cache)
        return npz["lh"].astype(np.float32), npz["rh"].astype(np.float32)

    # Last-resort fallback: synthesized icosphere (for tests / no-FS envs).
    import trimesh
    n_subdiv = 5 if source_space == "fs5" else 7
    sphere = trimesh.creation.icosphere(subdivisions=n_subdiv)
    coords_lh = sphere.vertices.astype(np.float32)
    return coords_lh, coords_lh.copy()


def extract_subject_fs5(
    recon_dir: Path,
    source_space: str = "fs5",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Extract per-vertex data for both hemispheres of one FreeSurfer subject.

    Auto-detects format:
    - fsaverage5 template (Phase A): reads .sulc/.curv/.thickness directly.
    - Remapped subject (Phase B): reads .fs5.sulc/.fs5.curv/.fs5.thickness.

    Parameters
    ----------
    recon_dir : Path
        FreeSurfer subjects dir entry, e.g.:
        - /usr/local/freesurfer/8.2.0/subjects/fsaverage5 (Phase A template)
        - $SUBJECTS_DIR/ixi002 (Phase B recon-all output, then remapped via Task A.6)
    source_space : str
        "fs5" (10242 vertices, default) or "fs7" (163842 vertices).

    Returns
    -------
    (lh_dict, rh_dict) : tuple of dict
        Each dict has keys 'sulc', 'curv', 'thickness', 'coords'.
        'coords' is loaded from the fsaverage{5,7} sphere via read_geometry.
    """
    surf_dir = Path(recon_dir) / "surf"
    formats = _detect_format(recon_dir)
    coords_lh, coords_rh = _load_fsaverage5_coords(source_space)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for hemi in ["lh", "rh"]:
        fmt = formats[hemi]
        if fmt == "fs5_template":
            d = {
                "sulc": read_morph_per_vertex(surf_dir / f"{hemi}.sulc"),
                "curv": read_morph_per_vertex(surf_dir / f"{hemi}.curv"),
                "thickness": read_morph_per_vertex(surf_dir / f"{hemi}.thickness"),
                "coords": coords_lh if hemi == "lh" else coords_rh,
            }
        else:  # fs5_remapped
            d = {
                "sulc": read_morph_per_vertex(surf_dir / f"{hemi}.fs5.sulc"),
                "curv": read_morph_per_vertex(surf_dir / f"{hemi}.fs5.curv"),
                "thickness": read_morph_per_vertex(surf_dir / f"{hemi}.fs5.thickness"),
                "coords": coords_lh if hemi == "lh" else coords_rh,
            }
        out[hemi] = d

    return out["lh"], out["rh"]


# Legacy alias (back-compat with code written before fs5_template/fs5_remapped
# auto-detection). Old callers that pass .mgh files still get .mgh handling.
def read_mgh_per_vertex(path: Path) -> np.ndarray:
    """Read a 1D per-vertex .mgh file as a 1D numpy array (legacy)."""
    return read_morph_per_vertex(path)