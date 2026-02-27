import torch
import torch.nn as nn
import numpy as np

class SEML(nn.Module):
    def __init__(self):
        super(SEML, self).__init__()
        self.meta_learner = nn.Linear(2, 1)
        
    def forward(self, b_gfg, b_soss):
        combined = torch.stack([b_gfg, b_soss], dim=-1)
        weights = self.meta_learner(combined)
        b_fused = torch.sum(weights * combined, dim=-1)
        return b_fused