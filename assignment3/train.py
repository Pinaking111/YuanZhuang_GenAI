import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Generator, Discriminator
import matplotlib.pyplot as plt
import numpy as np

def train(num_epochs=5, batch_size=64, lr=0.0002, beta1=0.5, device='cpu'):  # 减少轮数到5轮用于测试
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load MNIST dataset
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize generator and discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            label_real = torch.ones(batch_size, 1).to(device)
            label_fake = torch.zeros(batch_size, 1).to(device)

            # Train with real images
            output_real = discriminator(real_images)
            d_loss_real = criterion(output_real, label_real)

            # Train with fake images
            noise = torch.randn(batch_size, 100).to(device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(output_fake, label_fake)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            output_fake = discriminator(fake_images)
            g_loss = criterion(output_fake, label_real)
            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                      f'd_loss: {d_loss.item():.4f} g_loss: {g_loss.item():.4f}')

        # Save generated images every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_images(generator, epoch + 1, device)

    # Save the trained model
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict()
    }, 'gan_model.pt')

def save_images(generator, epoch, device, num_images=16):
    with torch.no_grad():
        noise = torch.randn(num_images, 100).to(device)
        generated_images = generator(noise).cpu()
        generated_images = generated_images * 0.5 + 0.5  # Denormalize

        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            if i < num_images:
                ax.imshow(generated_images[i].squeeze(), cmap='gray')
            ax.axis('off')
        plt.savefig(f'generated_images_epoch_{epoch}.png')
        plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device=device)