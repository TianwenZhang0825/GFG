import torch
import torch.nn as nn
import torch.nn.functional as F

def ICL_loss(pred, target, tau=0.5):
    iou = compute_iou(pred, target)
    loss = torch.mean(torch.clamp(iou - tau, min=0))
    return loss

def KLD_loss(pred_dist, target_dist):
    return F.kl_div(pred_dist.log(), target_dist, reduction='batchmean')

def UMDL_loss(pred, target, uncertainty):
    return F.mse_loss(pred, target) * torch.exp(-uncertainty) + uncertainty

def compute_iou(box1, box2):
    # Simplified IoU computation
    inter = torch.min(box1[..., 2], box2[..., 2]) - torch.max(box1[..., 0], box2[..., 0])
    inter = torch.clamp(inter, min=0)
    union = (box1[..., 2] - box1[..., 0]) + (box2[..., 2] - box2[..., 0]) - inter
    return inter / union