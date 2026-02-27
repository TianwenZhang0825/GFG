import torch
import torch.nn as nn
from models.pplvt import PPLVT
from models.pdet import PDet
from models.pccn import PCCN
from models.idet import IDet

class GFGBranch(nn.Module):
    def __init__(self):
        super(GFGBranch, self).__init__()
        self.pplvt = PPLVT()
        self.pdet = PDet()
        self.pccn = PCCN()
        self.idet = IDet()
        
    def forward(self, x):
        # Glance Panorama
        f_pano = self.pplvt(x)
        
        # Focus Population
        populations = self.pdet(f_pano)
        
        # Gaze Individual
        individuals = []
        for pop in populations:
            pop_context = self.pccn(pop)
            ind = self.idet(pop_context)
            individuals.append(ind)
        
        return torch.cat(individuals, dim=0)