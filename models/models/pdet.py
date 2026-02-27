import torch
import torch.nn as nn

class PDet(nn.Module):
    def __init__(self, in_channels=128):
        super(PDet, self).__init__()
        self.rpn = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 2, kernel_size=1)  # cls + reg
        )
        
    def forward(self, x):
        preds = self.rpn(x)
        cls_pred, reg_pred = preds[:, :1, :, :], preds[:, 1:, :, :]
        return cls_pred, reg_pred