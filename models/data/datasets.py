import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class SSDDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.images = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.jpg')]
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class HRSIDDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.images = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.png')]
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image