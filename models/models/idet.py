import torch
import torch.nn as nn

class IDet(nn.Module):
    def __init__(self):
        super(IDet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.cls_head = nn.Conv2d(512, 2, kernel_size=1)
        self.reg_head = nn.Conv2d(512, 4, kernel_size=1)
        
    def forward(self, x):
        features = self.backbone(x)
        cls = self.cls_head(features)
        reg = self.reg_head(features)
        return cls, reg