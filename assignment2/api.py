import io, json
import os
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from .model import TinyCifarCNN  # 使用相对导入

app = FastAPI()

# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "cnn_cifar10.pt")

# 加载模型
_ckpt = torch.load(model_path, map_location="cpu")
classes = _ckpt["classes"]
model = TinyCifarCNN(num_classes=len(classes))
model.load_state_dict(_ckpt["state_dict"])
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010)),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
    top = torch.topk(probs, k=1)
    label = classes[top.indices.item()]
    conf  = float(top.values.item())
    return {"label": label, "confidence": round(conf, 4)}