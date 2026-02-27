import torch
import torch.nn as nn

class PCCN(nn.Module):
    def __init__(self):
        super(PCCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(128, num_heads=4)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(2).transpose(1, 2)
        x, _ = self.attention(x, x, x)
        x = x.transpose(1, 2).reshape(x.shape[0], 128, x.shape[2] // 128, -1)
        return x