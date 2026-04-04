"""
Metabolic Saliency
==================
S_ms(i,t) = (d m_hat_i / d x_{i,t}) * C_TE^(i)

where C_TE^(i) = sum_{j != i} TE_{j->i}(l*) * exp(alpha * TE_{j->i}(l*))

Faithfulness: exact Jacobian via backpropagation (Proposition 3, Paper 1).
Stability: bounded under retraining (Corollary 1, Paper 1).
"""
import torch

def metabolic_saliency(model, x, te_matrix, alpha=1.0, sector_idx=None):
    """
    Compute Metabolic Saliency for all sectors or a specific sector.

    Args:
        model:      GWSSTNet + PMNet (callable: x -> m_hat)
        x:          (B, N, F, tau) input voxel, requires_grad=True
        te_matrix:  (N, N) transfer entropy matrix TE_{j->i}
        alpha:      temperature for TE weighting
        sector_idx: if None, compute for all N sectors

    Returns:
        S_ms: (B, N) Metabolic Saliency values
    """
    x = x.detach().requires_grad_(True)
    m_hat = model(x)          # (B, N)

    # C_TE^(i) = sum_{j!=i} TE_{j->i} * exp(alpha * TE_{j->i})
    te = te_matrix.clone()
    te.fill_diagonal_(0.0)
    C_TE = (te * torch.exp(alpha * te)).sum(dim=0)  # (N,)

    S_ms = torch.zeros_like(m_hat)
    sectors = [sector_idx] if sector_idx is not None else range(m_hat.shape[1])
    for i in sectors:
        grad = torch.autograd.grad(
            m_hat[:, i].sum(), x,
            retain_graph=True, create_graph=False)[0]  # (B, N, F, tau)
        jacobian_i = grad[:, i, :, :].norm(dim=(-2,-1))  # (B,) scalar per batch
        S_ms[:, i] = jacobian_i * C_TE[i]
    return S_ms
