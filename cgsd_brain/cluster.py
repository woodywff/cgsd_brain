"""cgsd_brain.cluster — curvature-aware spectral clustering (G3 algorithm).

Ported from cgsd_release/cgsd_release/cluster_strategies.py.

Algorithm:
1. Build k-NN graph on the encoder embedding (k=20 default).
2. Re-weight the k-NN graph by Forman-Ricci curvature:
   A'_ij = A_ij * sigmoid(α * κ_ij)
   where α controls curvature influence.
3. Run Ng-Jordan-Weiss spectral embedding: compute the top-K eigenvectors
   of the normalized graph Laplacian.
4. K-Means cluster in the eigenspace.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph


def cluster_kmeans(emb: np.ndarray, n_classes: int, seed: int = 0) -> np.ndarray:
    """Plain KMeans on the embedding."""
    km = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
    return km.fit_predict(emb)


def cluster_spectral(emb: np.ndarray, n_classes: int, seed: int = 0,
                     n_neighbors: int = 10) -> np.ndarray:
    """Spectral clustering on a kNN affinity built from the embedding.

    Uses Ng-Jordan-Weiss (symmetric normalized Laplacian) for stability.
    """
    A = kneighbors_graph(emb, n_neighbors=n_neighbors, mode="connectivity",
                         include_self=False, n_jobs=1)
    A = A.maximum(A.T)  # symmetrize
    A = A.tocsr()
    deg = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L = sp.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    L = L.tocsr()
    try:
        # k+1 smallest algebraic eigenvalues
        eigvals, eigvecs = spla.eigsh(L, k=n_classes + 1, which="SM",
                                       tol=1e-3)
    except Exception:
        # Fallback to dense for small graphs / numerical issues
        L_dense = L.toarray()
        w, v = np.linalg.eigh(L_dense)
        eigvals, eigvecs = w[:n_classes + 1], v[:, :n_classes + 1]
    # Drop the smallest (constant) eigenvector
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order[1:n_classes + 1]]
    eigvecs = eigvecs / (np.linalg.norm(eigvecs, axis=1, keepdims=True) + 1e-9)
    km = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
    return km.fit_predict(eigvecs)


def cluster_curvature_spectral(emb: np.ndarray, n_classes: int,
                                curvature: np.ndarray, edge_index: np.ndarray,
                                alpha: float = 1.0,
                                seed: int = 0,
                                n_neighbors: int = 10) -> np.ndarray:
    """G3: Spectral clustering with curvature-weighted embedding affinity.

    The original graph's edge curvature is projected onto the kNN graph
    of the embedding. This biases the spectral cut towards edges that
    are structurally 'tight' (positive curvature = within-community)
    and away from bridges (negative curvature = cross-community).

    Args:
        emb: [N, d] node embeddings.
        n_classes: K.
        curvature: [E] Forman-Ricci curvature per original-graph edge.
        edge_index: [2, E] long-format edge index of the original graph.
        alpha: multiplier on the curvature logit (alpha=0 -> vanilla
            spectral on kNN; alpha large -> bridge-suppressing).
        seed: KMeans RNG seed.
        n_neighbors: kNN graph degree.
    """
    # kNN affinity on the embedding
    A = kneighbors_graph(emb, n_neighbors=n_neighbors, mode="connectivity",
                         include_self=False, n_jobs=1)
    A = A.maximum(A.T)
    A = A.tocsr()

    # Build curvature logit for each (i, j) pair in the kNN graph
    coo = A.tocoo()
    src, dst = coo.row, coo.col
    # Build a dict from (u, v) -> curvature for the original graph
    curv_lookup = {}
    edge_index = np.asarray(edge_index)
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        curv_lookup[(u, v)] = float(curvature[k])
        curv_lookup[(v, u)] = float(curvature[k])

    curv_knn = np.array([
        curv_lookup.get((int(s), int(d)), 0.0) for s, d in zip(src, dst)
    ], dtype=np.float64)
    curv_logit = 1.0 / (1.0 + np.exp(-alpha * curv_knn))  # in (0, 1)
    A_curv = sp.coo_matrix((curv_logit, (src, dst)),
                           shape=A.shape).tocsr()
    A_curv = A_curv.maximum(A_curv.T)

    # Symmetric normalized Laplacian on the curvature-weighted affinity
    deg = np.asarray(A_curv.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L = sp.eye(A.shape[0]) - D_inv_sqrt @ A_curv @ D_inv_sqrt
    L = L.tocsr()
    try:
        eigvals, eigvecs = spla.eigsh(L, k=n_classes + 1, which="SM",
                                       tol=1e-3)
    except Exception:
        L_dense = L.toarray()
        w, v = np.linalg.eigh(L_dense)
        eigvals, eigvecs = w[:n_classes + 1], v[:, :n_classes + 1]
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order[1:n_classes + 1]]
    eigvecs = eigvecs / (np.linalg.norm(eigvecs, axis=1, keepdims=True) + 1e-9)
    km = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
    return km.fit_predict(eigvecs)
