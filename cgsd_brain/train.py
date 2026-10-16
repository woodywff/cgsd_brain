"""cgsd_brain.train — train the SheafDiffusion encoder on a single hemisphere.

Wraps cgsd_brain.sheaf + cgsd_brain.losses into a single
`train_cgsd_brain(...)` function.

For Phase 1 (MVP), we train one hemisphere at a time. The hemisphere
symmetry loss (when enabled) requires a second embedding Z_right which
the caller (run_all.py) computes separately and combines.

The output is the final embedding Z (N, out_features * n_heads).

NOTE — API mismatch with the original spec:
  * The actual SheafDiffusion signature is (in_features, out_features, n_heads, dropout)
    — `hidden_dim` is exposed to callers as a friendly alias and forwarded as
    `out_features` internally (in Task 8 we kept the canonical name from
    cgsd_release/sheaf.py).
  * `SheafDiffusion.forward` takes (x, edge_index, curvature=None), NOT
    (x, edge_index, num_nodes=...). The number of nodes is derived from
    x.shape[0] inside the model.
  * Loss functions are leading-underscored (private) in losses.py
    (`_modularity_loss`, `_collapse_loss_v2`, `_curvature_recon_loss`).
    We import them by their actual names.
  * `_modularity_loss` takes (S, edge_index, num_nodes), NOT (S, adj, num_nodes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from cgsd_brain.sheaf import SheafDiffusion, compute_forman_ricci_curvature
from cgsd_brain.losses import (
    _modularity_loss,
    _collapse_loss_v2,
    _curvature_recon_loss,
)


@dataclass
class TrainConfig:
    """All trainable hyperparameters in one place."""
    out_features: int = 64       # SheafDiffusion out_features (= hidden_dim)
    n_heads: int = 2
    dropout: float = 0.3
    lr: float = 0.005
    weight_decay: float = 5e-4
    n_epochs: int = 30
    w_mod: float = 1.0
    w_col: float = 5.0
    w_rec: float = 1.0
    w_sulcal: float = 1.0
    w_sym: float = 0.5
    sulcal_beta: float = 1.0
    n_clusters: int = 200        # Soft-assignment K (upper bound; cluster step picks actual K)
    seed: int = 0


def train_cgsd_brain(
    features: np.ndarray,        # (N, F) node features
    edge_index: np.ndarray,      # (2, E) COO edge index
    adj: sp.csr_matrix,          # (N, N) sparse adjacency (used by modularity, edge_index also passed)
    hidden_dim: int = 64,
    n_heads: int = 2,
    n_epochs: int = 30,
    lr: float = 0.005,
    w_mod: float = 1.0,
    w_col: float = 5.0,
    w_rec: float = 1.0,
    w_sulcal: float = 1.0,
    w_sym: float = 0.0,          # 0 = no symmetry term (single-hemi training)
    seed: int = 0,
) -> np.ndarray:
    """Train SheafDiffusion and return the final embedding.

    Parameters
    ----------
    features : np.ndarray, shape (N, F)
        Per-vertex node features (e.g., coords + sulc + curv + thickness).
    edge_index : np.ndarray, shape (2, E)
        COO format edge index.
    adj : csr_matrix, shape (N, N)
        Sparse adjacency (used by modularity loss).
    hidden_dim : int
        Per-head embedding dimension. The final output dim is
        ``hidden_dim * n_heads`` (the SheafDiffusion concatenates heads).
    n_heads : int
        Number of attention / projection heads.
    n_epochs : int
        Number of SGD epochs.
    lr : float
        Learning rate for Adam.
    w_mod : float
        Weight on the negative-modularity term.
    w_col : float
        Weight on the anti-collapse (balance) term.
    w_rec : float
        Weight on the curvature-reconstruction term.
    w_sulcal : float
        Accepted for API compatibility. In the MVP the sulcal weighting is
        baked into the affinity outside this function (or absent for the
        synthetic smoke test), so this term is a no-op inside the loop.
    w_sym : float
        Hemisphere-symmetry loss weight. 0 to disable (single-hemi training).
    seed : int
        RNG seed for torch + numpy.

    Returns
    -------
    Z : np.ndarray, shape (N, hidden_dim * n_heads)
        Final encoder embedding.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N, F = features.shape
    K = 200  # Default max K; cluster step picks the actual K

    # Compute Forman-Ricci curvature (used in recon loss and as sheaf edge weights)
    # Build the adjacency from edge_index for curvature calc.
    n_edges = edge_index.shape[1]
    curv_adj = sp.csr_matrix(
        (np.ones(n_edges, dtype=np.float32),
         (edge_index[0], edge_index[1])),
        shape=(N, N),
    )
    curvature = compute_forman_ricci_curvature(curv_adj)

    # Build model
    model = SheafDiffusion(
        in_features=F,
        out_features=hidden_dim,
        n_heads=n_heads,
        dropout=0.1,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    x = torch.from_numpy(features).float()
    ei = torch.from_numpy(edge_index).long()
    curv_t = torch.from_numpy(curvature).float()

    # Cluster projection head: maps per-head concatenated embedding -> K cluster scores.
    # Initialized once, then learned end-to-end with the rest of the model.
    proj = nn.Linear(hidden_dim * n_heads, K).to(x.device)
    opt.add_param_group({"params": proj.parameters()})

    for epoch in range(n_epochs):
        model.train()
        proj.train()
        opt.zero_grad()
        h = model(x, ei, curvature=curv_t)  # (N, hidden_dim * n_heads)
        # Soft assignment S: per-cluster affinity (N, K)
        logits = proj(h)
        S = torch.softmax(logits, dim=1)
        # Compute losses
        l_mod = _modularity_loss(S, ei, num_nodes=N)
        l_col = _collapse_loss_v2(S, mode="balance")
        l_rec = _curvature_recon_loss(S, curv_t, ei)
        # The sulcal term is a no-op for the MVP: the affinity weighting by
        # sulcal depth is applied upstream of training (in hemisphere.py
        # during run_all). w_sulcal is accepted as a parameter for API
        # compatibility with downstream callers.
        loss = w_mod * l_mod + w_col * l_col + w_rec * l_rec
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        Z = model(x, ei, curvature=curv_t)
    return Z.cpu().numpy()
