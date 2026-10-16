"""cgsd_brain.hemisphere — bilateral hemisphere symmetry loss.

Most cortical functions are bilateral (left and right hemispheres are
topologically similar). We exploit this by adding a loss term that
encourages the left and right hemisphere embeddings to agree on
vertices that share a group-template parcellation label.

The group-template parcellation is loaded from disk (e.g., Schaefer 2018
group template in fsaverage5 space). It is NEVER used for evaluation
(NMI/ARI/Dice are computed only against the test-time atlas), so the
algorithm remains unsupervised with respect to the evaluation metric.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def sample_label_agreement_pairs(
    gt_left: np.ndarray,    # (N_L,) int labels
    gt_right: np.ndarray,   # (N_R,) int labels
    n_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_pairs (i_L, i_R) pairs with high label agreement.

    Returns
    -------
    pairs : np.ndarray, shape (n_pairs, 2)
        First column = indices into left hemisphere, second = right.
    """
    N_L, N_R = len(gt_left), len(gt_right)
    # Build label-agreement matrix (N_L, N_R): 1 if labels match, else 0
    agreement = (gt_left[:, None] == gt_right[None, :]).astype(np.float32)
    # Flatten and sample with probability ∝ agreement
    probs = agreement.flatten()
    probs = probs / probs.sum()  # normalize
    flat_idx = rng.choice(N_L * N_R, size=n_pairs, p=probs, replace=True)
    i_L = flat_idx // N_R
    i_R = flat_idx % N_R
    return np.stack([i_L, i_R], axis=1)


def hemisphere_symmetry_loss(
    Z_left: torch.Tensor,        # (N_L, d) left hemi embedding
    Z_right: torch.Tensor,       # (N_R, d) right hemi embedding
    gt_left: np.ndarray,         # (N_L,) group-template parcellation labels
    gt_right: np.ndarray,        # (N_R,) group-template parcellation labels
    pairs: np.ndarray,           # (P, 2) pre-sampled index pairs
) -> torch.Tensor:
    """MSE between Z_left[i_L] and Z_right[i_R] for label-matched pairs.

    The caller (train.py) is responsible for sampling pairs once per
    epoch using `sample_label_agreement_pairs`. We do not sample here
    so the loss is deterministic per epoch (given fixed pairs).
    """
    i_L = torch.as_tensor(pairs[:, 0], dtype=torch.long, device=Z_left.device)
    i_R = torch.as_tensor(pairs[:, 1], dtype=torch.long, device=Z_right.device)
    return F.mse_loss(Z_left[i_L], Z_right[i_R])