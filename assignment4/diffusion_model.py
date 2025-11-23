import torch
import torch.nn as nn

# Minimal UNet-like block for demo purposes (not a full production model)
class SimpleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.enc1 = SimpleBlock(in_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels*2, 4, stride=2, padding=1)
        self.enc2 = SimpleBlock(base_channels*2, base_channels*2)
        self.down2 = nn.Conv2d(base_channels*2, base_channels*4, 4, stride=2, padding=1)
        self.enc3 = SimpleBlock(base_channels*4, base_channels*4)

        self.mid = SimpleBlock(base_channels*4, base_channels*4)

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1)
        self.dec2 = SimpleBlock(base_channels*4, base_channels*2)
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1)
        self.dec1 = SimpleBlock(base_channels*2, base_channels)

        self.out = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        d1 = self.enc2(self.down1(e1))
        d2 = self.enc3(self.down2(d1))
        m = self.mid(d2)
        u2 = self.up2(m)
        cat2 = torch.cat([u2, d1], dim=1)
        dec2 = self.dec2(cat2)
        u1 = self.up1(dec2)
        cat1 = torch.cat([u1, e1], dim=1)
        dec1 = self.dec1(cat1)
        return self.out(dec1)

# Helper: sinusoidal time embedding (same formula as theory)
def sinusoidal_embedding(timesteps, dim):
    device = timesteps.device
    half = dim // 2
    emb = torch.log(torch.tensor(10000.0)) * torch.arange(half, dtype=torch.float32, device=device) / half
    emb = torch.exp(-emb)
    emb = timesteps[:, None].float() * emb[None, :]
    sin = torch.sin(emb)
    cos = torch.cos(emb)
    return torch.cat([sin, cos], dim=1)
