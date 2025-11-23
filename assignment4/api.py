import os
import io
import torch
from fastapi import FastAPI, Response
from PIL import Image
import torchvision.transforms as T

app = FastAPI(title="Assignment4 API", description="Diffusion and EBM demo endpoints")

# Paths
CHECKPOINT_DIR = '/app/assignment4/checkpoints' if os.path.exists('/app') else 'assignment4/checkpoints'
DIFF_PATH = os.path.join(CHECKPOINT_DIR, 'diffusion_model.pt')
EBM_PATH = os.path.join(CHECKPOINT_DIR, 'energy_model.pt')

@app.get('/generate/diffusion')
async def gen_diffusion():
    if not os.path.exists(DIFF_PATH):
        return {"error": "diffusion checkpoint not found. Run assignment4/train_diffusion.py"}
    # Minimal placeholder: return an empty image
    img = Image.new('RGB', (32,32), color=(73,109,137))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return Response(content=buf.getvalue(), media_type='image/png')

@app.get('/generate/ebm')
async def gen_ebm():
    if not os.path.exists(EBM_PATH):
        return {"error": "energy model checkpoint not found. Run assignment4/train_energy.py"}
    img = Image.new('RGB', (32,32), color=(200,100,50))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return Response(content=buf.getvalue(), media_type='image/png')

@app.get('/')
async def root():
    return {"message":"Assignment4 API. Endpoints: /generate/diffusion , /generate/ebm"}
