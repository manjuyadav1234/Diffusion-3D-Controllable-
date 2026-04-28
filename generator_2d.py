import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator2D(nn.Module):
    def __init__(self, input_channels=5, output_channels=3, latent_dim=128):
        super(Generator2D, self).init()
        # Initial projection of background latent code
        self.bg_fc = nn.Linear(latent_dim, 4 * 4 * 256)
        
        # Encoder for projected features
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        
        # Decoder
        self.dec3 = self._up_block(512, 128) # 256 (enc) + 256 (bg)
        self.dec2 = self._up_block(128 + 128, 64)
        self.dec1 = self._up_block(64 + 64, 32)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_map, bg_code):
        # Process background
        bg = self.bg_fc(bg_code).view(-1, 256, 4, 4)
        bg = F.interpolate(bg, size=feat_map.shape[2:], mode='bilinear')
        
        # U-Net flow
        e1 = self.enc1(feat_map)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Combine background and features
        d3 = self.dec3(torch.cat([e3, bg], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return self.final_conv(d1)