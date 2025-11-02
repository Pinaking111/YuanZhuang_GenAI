import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Initial fully connected layer to 7x7x128
        self.fc = nn.Linear(100, 7 * 7 * 128)
        
        # Transposed convolution layers
        self.conv_layers = nn.Sequential(
            # First transposed conv layer: 128 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Second transposed conv layer: 64 -> 1
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # x shape: (batch_size, 100)
        x = self.fc(x)
        # Reshape to (batch_size, 128, 7, 7)
        x = x.view(-1, 128, 7, 7)
        # Apply transposed convolutions
        x = self.conv_layers(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv layer: 1 -> 64
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second conv layer: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Flatten and apply linear layer
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        x = self.conv_layers(x)
        # Flatten
        x = x.view(-1, 128 * 7 * 7)
        # Apply final linear layer and sigmoid
        x = self.fc(x)
        x = self.sigmoid(x)
        return x