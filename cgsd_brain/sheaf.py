"""SheafDiffusion — label-free architecture for CGSD-Brain.

Ported verbatim from cgsd_release/cgsd_release/sheaf.py (which itself is
a copy of the canonical experiments/sheaf_curvature_001/stage-22/code/main.py
architecture — the supervised loss wrapper is NOT copied).

Used by cgsd_brain.train and cgsd_brain.run_all.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.utils import degree


def compute_forman_ricci_curvature(adj) -> np.ndarray:
    """Discrete Forman-Ricci curvature per edge.

    F(e) = 4 - deg(u) - deg(v) for unweighted graphs.
    Negative curvature = bridge/cut edges; positive = dense local clusters.
    """
    if sp.issparse(adj):
        deg = np.array(adj.sum(axis=1)).flatten()
        coo = adj.tocoo()
    else:
        deg = adj.sum(axis=1)
        coo = sp.coo_matrix(adj).tocoo()

    curvature = 4.0 - deg[coo.row] - deg[coo.col]
    return curvature


class SheafDiffusion(nn.Module):
    """Curvature-guided sheaf diffusion with self-loops and residual connection.

    Pure torch; no labels.
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int = 2, dropout: float = 0.5):
        super().__init__()
        self.n_heads = n_heads
        self.head_projections = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False)
            for _ in range(n_heads)
        ])
        self.residual = nn.Linear(in_features, out_features * n_heads, bias=False)
        self._init_weights()

    def _init_weights(self):
        for proj in self.head_projections:
            nn.init.xavier_uniform_(proj.weight, gain=1.414)
        nn.init.xavier_uniform_(self.residual.weight, gain=1.414)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                curvature: torch.Tensor | np.ndarray | None = None) -> torch.Tensor:
        """Apply curvature-guided sheaf diffusion.

        Args:
            x: [N, in_features]
            edge_index: [2, E]
            curvature: [E] edge-wise curvature values
        Returns:
            [N, out_features * n_heads]
        """
        N = x.shape[0]
        row, col = edge_index
        E = row.shape[0]

        # Degree normalization
        deg = degree(row, N, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[~torch.isfinite(deg_inv_sqrt)] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [E]

        # Curvature-guided weighting
        if curvature is not None:
            curv = curvature if isinstance(curvature, torch.Tensor) else torch.tensor(
                curvature, dtype=x.dtype, device=x.device
            )
            # Positive curvature -> within-community -> higher weight
            # Negative curvature -> cross-community -> lower weight
            curv_norm = torch.sigmoid(curv)
            edge_w = norm * curv_norm
        else:
            edge_w = norm

        head_outputs = []
        for h in range(self.n_heads):
            h_x = self.head_projections[h](x)
            msg = h_x[col] * edge_w.unsqueeze(1)
            out_h = torch.zeros_like(h_x)
            out_h.index_add_(0, row, msg)
            head_outputs.append(out_h)

        out = torch.cat(head_outputs, dim=1)  # [N, out * n_heads]

        # Residual + self-loop boost
        residual = self.residual(x)
        out = 0.5 * out + 0.5 * residual

        return out
