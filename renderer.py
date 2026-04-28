import torch
import torch.nn as nn

class DifferentiableRenderer(nn.Module):
    def __init__(self, img_size=64, n_primitives=5, sigma=0.05):
        super().__init__()
        self.img_size = img_size
        self.n_primitives = n_primitives
        self.sigma = sigma

    def project_primitives(self, translation, scale, camera):
        azimuth, elevation, distance, focal = camera[:, 0:1], camera[:, 1:2], camera[:, 2:3].clamp(min=1.0), camera[:, 3:4].clamp(min=0.5)
        cos_a, sin_a, cos_e, sin_e = torch.cos(azimuth), torch.sin(azimuth), torch.cos(elevation), torch.sin(elevation)
        
        tx, ty, tz = translation[..., 0], translation[..., 1], translation[..., 2]
        cam_x = cos_a * tx - sin_a * tz
        cam_y = sin_e * (sin_a * tx + cos_a * tz) + cos_e * ty
        cam_z = (cos_e * (sin_a * tx + cos_a * tz) - sin_e * ty + distance).clamp(min=0.1)
        
        return focal * cam_x / cam_z, focal * cam_y / cam_z, focal.unsqueeze(-1) * scale[..., :2] / cam_z.unsqueeze(-1)

    def render_soft_boxes(self, proj_x, proj_y, proj_s, existence):
        B, N = proj_x.shape
        H = W = self.img_size
        xs = torch.linspace(-1, 1, W, device=proj_x.device)
        ys = torch.linspace(-1, 1, H, device=proj_x.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        
        cx, cy = proj_x.view(B, N, 1, 1), proj_y.view(B, N, 1, 1)
        sw, sh = proj_s[:, :, 0].view(B, N, 1, 1).clamp(min=0.01), proj_s[:, :, 1].view(B, N, 1, 1).clamp(min=0.01)
        
        feat = torch.exp(-0.5 * (((grid_x - cx) / (sw + self.sigma))**2 + ((grid_y - cy) / (sh + self.sigma))**2))
        return feat * existence.view(B, N, 1, 1)

    def forward(self, primitives, camera):
        proj_x, proj_y, proj_s = self.project_primitives(primitives['translation'], primitives['scale'], camera)
        return self.render_soft_boxes(proj_x, proj_y, proj_s, primitives['existence'].squeeze(-1))