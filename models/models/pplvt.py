import torch
import torch.nn as nn

class PPLVT(nn.Module):
    def __init__(self, patch_size=4, dim=128, num_heads=4):
        super(PPLVT, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, dim, kernel_size=3, stride=2)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=256),
            num_layers=4
        )
        
    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x