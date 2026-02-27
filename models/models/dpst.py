import torch
import torch.nn as nn
from models.gfg import GFGBranch
from models.soss import SOSSBranch
from models.seml import SEML

class DPST(nn.Module):
    def __init__(self):
        super(DPST, self).__init__()
        self.gfg = GFGBranch()
        self.soss = SOSSBranch()
        self.seml = SEML()
        
    def forward(self, x):
        b_gfg = self.gfg(x)
        b_soss = self.soss(x)
        b_final = self.seml(b_gfg, b_soss)
        return b_final