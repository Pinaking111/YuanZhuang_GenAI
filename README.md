# SPS GenAI Assignments

This repository contains multiple assignment demos and small APIs. The current focus is
Assignment 5 — an LLM fine-tuning demo using a small GPT-2 model and a text-generation API.

Supported assignments in this repo (examples):
- Assignment 2: CNN for CIFAR-10 (legacy)
- Assignment 3: GAN for MNIST (legacy)
- Assignment 5: LLM fine-tuning demo (current)

## Requirements

- Docker (optional, recommended for reproducible runs)
- Python 3.12+ and a virtual environment (recommended: `.venv`)

## Quick Start — Assignment 5 (recommended)

1) Build the Docker image (optional):

```bash
docker build -t sps-genai:assignment5 .
```

2) Run the container (optional):

```bash
docker run -p 8000:8000 sps-genai:assignment5
```

3) Or run locally in the project's virtualenv (fastest for development):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install transformers datasets accelerate sentencepiece torch uvicorn fastapi python-multipart numpy pillow matplotlib

# (Optional) run a quick smoke fine-tune to create a local checkpoint
python assignment5/fine_tune.py --split "train[:100]" --num_train_epochs 1 --per_device_train_batch_size 1

# start the API
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 &

# call the generate endpoint (example)
curl -X POST http://localhost:8000/assignment5/generate \
    -H "Content-Type: application/json" \
    -d '{"model_dir":"assignment5/checkpoint","prompt":"Hello, how are you?","max_new_tokens":50}'
```

## What Assignment 5 provides

- `assignment5/fine_tune.py`: fine-tuning script (Hugging Face Trainer). Saves a checkpoint to `assignment5/checkpoint/`.
- `main.py` (root): FastAPI app exposing `POST /assignment5/generate` which loads a tokenizer + model and returns generated text.
- `assignment5/part2_answers.md`: theory answers for the reinforcement-learning questions.

## Submission notes

- For graders: include `assignment5/` (training script, main API, README, theory answers). If you include `assignment5/checkpoint/`, note it may be large — consider providing a zipped checkpoint or instructions to reproduce using `fine_tune.py`.

## Legacy assignments

Older assignment folders remain for reference (`assignment2/`, `assignment3/`). They are not required for Assignment 5 grading.

