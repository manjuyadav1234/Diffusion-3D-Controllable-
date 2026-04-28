import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).init()
        
        def critic_block(in_c, out_c, stride=2, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            # Spectral Norm for training stability
            return nn.Sequential(*[spectral_norm(layer) if isinstance(layer, nn.Conv2d) else layer for layer in layers])

        self.model = nn.Sequential(
            critic_block(input_channels, 64, normalize=False), # 32x32
            critic_block(64, 128),                             # 16x16
            critic_block(128, 256),                            # 8x8
            critic_block(256, 512, stride=1),                  # 7x7
            nn.Conv2d(512, 1, 4, padding=1)                    # 6x6 Patch output
        )

    def forward(self, x):
        return self.model(x)