import io
import torch
from fastapi import FastAPI, Response
from PIL import Image
import matplotlib.pyplot as plt
from model import Generator

app = FastAPI()

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("gan_model.pt", map_location=device)
generator = Generator().to(device)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

@app.post("/generate_digit")
async def generate_digit():
    # Generate a random noise vector
    noise = torch.randn(1, 100).to(device)
    
    # Generate image
    with torch.no_grad():
        generated_image = generator(noise)
        # Denormalize the image
        generated_image = generated_image * 0.5 + 0.5
        
    # Convert to PIL Image
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    plt.imshow(generated_image[0].cpu().squeeze(), cmap='gray')
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    
    return Response(content=buf.getvalue(), media_type="image/png")