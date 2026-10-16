"""Structural losses for CGSD-Brain: modularity, anti-collapse, curvature recon.

Ported verbatim from cgsd_release/cgsd_release/losses.py. No brain-specific
additions (those are in cgsd_brain.hemisphere).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.utils import degree


def _modularity_loss(S: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Negative modularity of the soft cluster assignment."""
    row, col = edge_index
    S = torch.clamp(S, min=1e-10, max=1.0)
    m = row.numel() / 2.0 + 1e-9
    deg = degree(row, num_nodes, dtype=S.dtype)
    d_c = S.t() @ deg / (2 * m)
    s_u, s_v = S[row], S[col]
    e_c = (s_u * s_v).sum(dim=0) / m
    Q = (e_c - d_c ** 2).sum()
    return -Q


def _collapse_loss_v2(S: torch.Tensor, mode: str = "balance") -> torch.Tensor:
    """Multiple anti-collapse modes.

    Args:
        S: [N, K] soft cluster assignment (rows sum to 1).
        mode: one of {"balance", "uniform", "entropy", "minmax"}.
            - balance: orthogonality (S^T S ≈ (N/K) I)
            - uniform: encourage uniform cluster occupancy
            - entropy: maximize per-row entropy
            - minmax: penalize cluster-size CV² directly
    """
    K = S.shape[1]
    gram = S.t() @ S  # (K, K)
    N = S.shape[0]
    if mode == "balance":
        target = (N / K) * torch.eye(K, device=S.device, dtype=S.dtype)
        return ((gram - target) ** 2).sum() / (N ** 2)
    if mode == "uniform":
        cluster_size = S.sum(0)
        target_size = N / K
        return ((cluster_size - target_size) ** 2).mean() / (N ** 2 / K ** 2)
    if mode == "entropy":
        ent = -(S * torch.log(S + 1e-10)).sum(dim=1).mean()
        return -ent
    if mode == "minmax":
        cluster_frac = S.sum(0) / N
        return ((cluster_frac.std() / (cluster_frac.mean() + 1e-9)) ** 2)
    raise ValueError(f"Unknown mode: {mode}")


def _curvature_recon_loss(S: torch.Tensor, curvature: torch.Tensor,
                          edge_index: torch.Tensor) -> torch.Tensor:
    """MSE between predicted edge cluster agreement and sigmoid(curvature)."""
    row, col = edge_index
    curv = curvature if isinstance(curvature, torch.Tensor) else torch.tensor(
        curvature, dtype=S.dtype, device=S.device
    )
    target = torch.sigmoid(curv)
    s_u, s_v = S[row], S[col]
    pred = (s_u * s_v).sum(dim=-1)
    return F.mse_loss(pred, target)
