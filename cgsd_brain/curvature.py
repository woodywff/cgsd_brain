"""cgsd_brain.curvature — Forman-Ricci + Ollivier-Ricci curvature on graphs.

Forman-Ricci curvature is the discrete analog of Ricci curvature for
weighted cell complexes. For an unweighted graph:
    κ_e = 4 - deg(u) - deg(v)
where deg(u) is the degree of vertex u (number of edges incident to u).

Ollivier-Ricci curvature uses the Wasserstein-1 distance between lazy
random walk distributions on the neighborhoods of u and v:
    κ(u, v) = 1 - W_1(μ_u, μ_v) / d(u, v)

Reference: Sreejith et al. (2016) "Forman-Ricci Curvature for Networks".
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def compute_forman_ricci(
    edge_index: np.ndarray,  # (2, E) where edge_index[0] = u, edge_index[1] = v
    num_nodes: int,
) -> np.ndarray:
    """Compute Forman-Ricci curvature per edge.

    κ_e = 4 - deg(u) - deg(v)

    Parameters
    ----------
    edge_index : np.ndarray, shape (2, E)
        Edge index in COO format. Each column is one edge.
    num_nodes : int
        Total number of nodes in the graph.

    Returns
    -------
    curvature : np.ndarray, shape (E,)
        Per-edge Forman-Ricci curvature.
    """
    assert edge_index.shape[0] == 2, "edge_index must be (2, E)"
    u, v = edge_index[0], edge_index[1]
    # Compute degree of each endpoint
    deg = np.bincount(u, minlength=num_nodes).astype(np.int64)
    deg += np.bincount(v, minlength=num_nodes).astype(np.int64)
    curvature = 4.0 - deg[u] - deg[v]
    return curvature.astype(np.float32)


def forman_to_edge_dict(
    edge_index: np.ndarray,
    curvature: np.ndarray,
) -> Dict[Tuple[int, int], float]:
    """Map (u, v) tuple → curvature for fast lookup."""
    return {
        (int(u), int(v)): float(c)
        for u, v, c in zip(edge_index[0], edge_index[1], curvature)
    }


def compute_ollivier_ricci(
    edge_index: np.ndarray,
    num_nodes: int,
    alpha: float = 0.5,
    n_iters: int = 20,
) -> np.ndarray:
    """Compute Ollivier-Ricci curvature (lazy random walk + Sinkhorn).

    NOTE: Stub for Phase 1 MVP. Full implementation is a Phase 3 ablation.

    For now returns zeros of the right shape, so the code path exists.
    """
    return np.zeros(edge_index.shape[1], dtype=np.float32)