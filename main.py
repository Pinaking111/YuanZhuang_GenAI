from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# Root app focused only on Assignment 5 per user request.
app = FastAPI(title="SPS GenAI - Assignment 5",
              description="Server exposing a small Assignment 5 text-generation endpoint")


@app.get('/', include_in_schema=False)
async def root():
    return {"message": "Assignment5 endpoints mounted. Use POST /assignment5/generate to run generation."}


class GenRequest(BaseModel):
    model_dir: str | None = "assignment5/checkpoint"
    prompt: str | None = None
    max_new_tokens: int | None = 64


@app.post('/assignment5/generate')
async def assignment5_generate(req: GenRequest):
    """Generate text using a (fine-tuned) GPT-2 model located at model_dir.

    Loads tokenizer and model lazily so server boots even if transformers is not installed.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"transformers/torch not available: {e}")

    model_dir = req.model_dir or 'assignment5/checkpoint'
    prompt = req.prompt or "Question: What is AI? Context: AI stands for artificial intelligence.\nAnswer: "

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to load model from {model_dir}: {e}")

    inputs = tokenizer(prompt, return_tensors='pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=int(req.max_new_tokens), do_sample=True, top_k=50, top_p=0.95)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": text}
