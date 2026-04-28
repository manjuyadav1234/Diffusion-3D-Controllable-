import torch
import torch.nn as nn
import numpy as np

class PrimitiveGenerator(nn.Module):
    def __init__(self, latent_dim=128, n_primitives=5, hidden_dim=512):
        super().__init__()
        self.n_primitives = n_primitives
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.primitive_head = nn.Linear(hidden_dim, n_primitives * 10)
        self.bg_head = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, z):
        B = z.shape[0]
        h = self.encoder(z)
        raw = self.primitive_head(h).view(B, self.n_primitives, 10)
        
        primitives = {
            'translation': torch.tanh(raw[..., 0:3]),
            'scale': torch.sigmoid(raw[..., 3:6]) * 0.5 + 0.05,
            'rotation': torch.tanh(raw[..., 6:9]) * np.pi,
            'existence': torch.sigmoid(raw[..., 9:10]),
        }
        bg_code = torch.tanh(self.bg_head(h))
        return primitives, bg_code