import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class CLIPLoss(nn.Module):
    def __init__(self, model_name='ViT-B/32', device='cuda'):
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self.device = device
    
    def forward(self, images, text_prompts):
        imgs = F.interpolate(images, size=(224, 224), mode='bilinear')
        imgs = (imgs + 1) / 2.0
        # Standard CLIP normalization constants
        mean = torch.tensor([0.4814, 0.4578, 0.4082], device=images.device).view(1,3,1,1)
        std  = torch.tensor([0.2686, 0.2613, 0.2757], device=images.device).view(1,3,1,1)
        imgs = (imgs - mean) / std
        
        img_feat = F.normalize(self.model.encode_image(imgs).float(), dim=-1)
        text_feat = F.normalize(clip.tokenize(text_prompts).to(self.device), dim=-1)
        
        sim = img_feat @ text_feat.T
        return 1.0 - sim.max(dim=-1).values.mean()

def evaluate_3d_consistency(model, n_samples=50):
    """
    3D Consistency Score:
    For the same z, render from 2 nearby viewpoints.
    High consistency = low perceptual distance between close views.
    """
    model.eval()
    diffs = []
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    for _ in range(n_samples):
        az1 = 0.0
        az2 = 0.1  # small 5.7° rotation
        
        cam1 = model.sample_camera(1, DEVICE, fixed_azimuth=az1)
        cam2 = model.sample_camera(1, DEVICE, fixed_azimuth=az2)
        
        img1, _ = model.generate(z, cam1)
        img2, _ = model.generate(z, cam2)
        
        d = lpips_fn(img1, img2).item()
        diffs.append(d)
    
    return np.mean(diffs), np.std(diffs)