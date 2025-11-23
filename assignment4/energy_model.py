import torch
import torch.nn as nn

class SimpleEBM(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base, base*2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base*2, base*4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base*4, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)
