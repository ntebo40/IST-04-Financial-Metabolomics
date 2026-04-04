"""
PMNet: Power Mapping Network
============================
Maps GWS-STNet bottleneck Z^(4) -> metabolic power scores m_hat in R^N.
All weight matrices spectrally normalised (||W||_2 <= 1).
GELU activation ensures Jacobian exists everywhere (required for Metabolic Saliency).
"""
import torch.nn as nn

class PMNet(nn.Module):
    def __init__(self, d4=768, N=87):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(d4, 512)), nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(512, 256)), nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(256, N)))

    def forward(self, z):
        """z: (B, N, d4) -> m_hat: (B, N)"""
        return self.net(z).squeeze(-1) if z.shape[-1] != z.shape[-2] \
               else self.net(z).diagonal(dim1=-2, dim2=-1)
