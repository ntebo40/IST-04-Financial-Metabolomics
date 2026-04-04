"""
GWS-STNet: Gaussian-Weighted Swin Spatio-Temporal Network
=========================================================
Financial Metabolomics Series, Paper 1
Author: Prof. Ntebogang Dinah Moroke (ORCID: 0000-0001-8545-1860)
North-West University, South Africa
DOI: https://doi.org/10.5281/zenodo.19072906
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np

def gaussian_kernel(window_size, sigma, device):
    """Row-normalised Gaussian attention mask K_sigma (M^2 x M^2)."""
    assert sigma < 1.0/(2*np.pi)**0.5, \
        f"sigma={sigma:.4f} violates contractivity condition sigma < sigma*={1/(2*np.pi)**0.5:.4f}"
    c = torch.arange(window_size, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(c, c, indexing='ij')
    pos = torch.stack([gy.flatten(), gx.flatten()], -1)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    K = torch.exp(-(diff**2).sum(-1) / (2*sigma**2))
    return K / K.sum(-1, keepdim=True)

class GaussianWindowAttention(nn.Module):
    """Spectrally-normalised Gaussian-weighted Swin attention."""
    def __init__(self, dim, window_size, num_heads, sigma=0.39):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.sigma     = sigma
        self.window_size = window_size
        self.qkv  = nn.utils.spectral_norm(nn.Linear(dim, dim*3))
        self.proj = nn.utils.spectral_norm(nn.Linear(dim, dim))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2,0,3,1,4).unbind(0)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        K_s  = gaussian_kernel(self.window_size, self.sigma, x.device)
        attn = F.softmax(attn * K_s, dim=-1)
        return self.proj((attn @ v).transpose(1,2).reshape(B,N,C))

class GWSTransformerBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads, sigma=0.39, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = GaussianWindowAttention(dim, window_size, num_heads, sigma)
        self.norm2 = nn.LayerNorm(dim)
        m = int(dim*mlp_ratio)
        self.ffn = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(dim, m)), nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(m, dim)))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))

class GWSSTNet(nn.Module):
    """4-stage hierarchical GWS-STNet encoder. Input: (B,N,F,tau) -> (B,N,d4)"""
    def __init__(self, N=87, F=3, tau=22, sigma=0.39, d0=96,
                 depths=(2,2,6,2), num_heads=(3,6,12,24), window_size=7):
        super().__init__()
        self.patch_embed = nn.utils.spectral_norm(nn.Linear(F, d0))
        self.norm0 = nn.LayerNorm(d0)
        dims = [d0*(2**i) for i in range(4)]
        self.blocks = nn.ModuleList()
        self.merges  = nn.ModuleList()
        for i in range(4):
            self.blocks.append(nn.Sequential(*[
                GWSTransformerBlock(dims[i], window_size, num_heads[i], sigma)
                for _ in range(depths[i])]))
            if i < 3:
                self.merges.append(nn.Sequential(
                    nn.LayerNorm(dims[i]),
                    nn.utils.spectral_norm(nn.Linear(dims[i], dims[i+1]))))
        self.norm_out = nn.LayerNorm(dims[-1])

    def forward(self, x):
        B, N, F, T = x.shape
        x = self.norm0(self.patch_embed(x.reshape(B*N, T, F))).mean(1)
        x = x.reshape(B, N, -1)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i < 3:
                x = self.merges[i](x)
        return self.norm_out(x)
