"""cgsd_brain.sulcal — sulcal-depth-weighted affinity matrix.

The sulcal depth (FreeSurfer `sulc` measure) is a per-vertex scalar in
[-1, +1] that quantifies how "deep" a vertex sits in a cortical fold
(negative) vs on a gyrus top (positive). The Desikan-Killiany atlas was
hand-annotated by following sulcal fundi, so sulcal depth is a strong
geometric prior for parcellation boundaries.

We combine it with geometric kNN distance into a joint affinity:

    A_ij = exp(-d_ij² / σ²) · exp(-β · |sulc_i - sulc_j|)

When β=0, this reduces to pure geometric kNN affinity (vanilla CGSD).
When β→∞, only edges that DON'T cross large sulcal-depth differences
(i.e., edges that stay within a sulcal fundus or within a gyrus) survive.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def build_sulcal_affinity(
    coords: np.ndarray,        # (N, 3) fsaverage5 3D coords
    sulc: np.ndarray,         # (N,) FreeSurfer sulc measure, normalized to [-1, +1]
    k: int = 20,               # k-NN parameter
    beta: float = 1.0,         # sulcal weighting strength
) -> csr_matrix:
    """Build a sulcal-depth-weighted kNN affinity matrix.

    A_ij = exp(-d_ij² / σ²) · exp(-β · |sulc_i - sulc_j|)

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        3D coordinates in fsaverage5 space.
    sulc : np.ndarray, shape (N,)
        Per-vertex sulcal depth (FreeSurfer `sulc`).
    k : int
        Number of nearest neighbors per vertex.
    beta : float
        Sulcal weighting strength. β=0 → pure geometric. β→∞ → only
        edges with similar sulc depth survive.

    Returns
    -------
    A : csr_matrix, shape (N, N)
        Sparse symmetric affinity matrix.
    """
    N = coords.shape[0]
    assert sulc.shape == (N,), f"sulc must have shape (N,); got {sulc.shape}"
    assert k < N, f"k={k} must be < N={N}"

    # Step 1: Find k-NN
    knn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(coords)
    distances, neighbors = knn.kneighbors(coords)

    # Step 2: Compute geometric weight
    sigma = distances.std() + 1e-8  # avoid div by zero
    geo_weight = np.exp(-(distances ** 2) / (2 * sigma ** 2))

    # Step 3: Compute sulcal weight
    sulc_neighbor = sulc[neighbors]  # (N, k)
    sulc_diff = np.abs(sulc_neighbor - sulc[:, None])  # (N, k)
    sulcal_weight = np.exp(-beta * sulc_diff)

    # Step 4: Combine
    weights = (geo_weight * sulcal_weight).astype(np.float32)

    # Step 5: Build sparse symmetric matrix
    rows = np.repeat(np.arange(N), k)
    cols = neighbors.flatten()
    data = weights.flatten()
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    # Symmetrize: max(A, A.T)
    A = A.maximum(A.T)
    return A
