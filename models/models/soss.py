import torch
import torch.nn as nn
from models.padss import PADSS
from models.pgqr import PGQR
from models.pgam import PGAM

class SOSSBranch(nn.Module):
    def __init__(self):
        super(SOSSBranch, self).__init__()
        self.backbone = ResNet50()
        self.padss = PADSS()
        self.pgqr = PGQR()
        self.pgam = PGAM()
        
    def forward(self, x, skf, populations):
        features = self.backbone(x)
        queries = self.padss(features, populations)
        refined_queries = self.pgqr(queries, skf)
        detections = self.pgam(refined_queries, populations)
        return detections