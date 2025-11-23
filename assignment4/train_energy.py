import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from energy_model import SimpleEBM
import os

def get_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)

# Simple training loop that minimizes energy for real samples and maximizes for random noise
def train(epochs=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleEBM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = get_dataloader()

    for epoch in range(epochs):
        for xb, _ in loader:
            xb = xb.to(device)
            noise = torch.randn_like(xb)
            e_real = model(xb).mean()
            e_fake = model(noise).mean()
            loss = e_real - e_fake
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch} loss {loss.item():.4f}")
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'state_dict': model.state_dict()}, 'checkpoints/energy_model.pt')

if __name__ == '__main__':
    train(epochs=1)
