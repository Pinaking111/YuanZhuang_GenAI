# SPS GenAI Assignments

This repository contains two machine learning models with REST APIs:
1. Assignment 2: CNN implementation for CIFAR10 image classification
2. Assignment 3: GAN implementation for generating handwritten digits

## Requirements

- Docker
- Python 3.12+ (if running locally)

## Quick Start with Docker

1. Build the Docker image:
```bash
docker build -t sps_genai .
```

2. Run the container:
```bash
docker run -p 8000:8000 -p 8001:8001 sps_genai
```

3. Access the APIs:
- Assignment 2 (CIFAR10): http://localhost:8000/docs
- Assignment 3 (GAN): http://localhost:8001/docs

## Local Development

1. Install dependencies:
```bash
pip install uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

2. Train the models:
```bash
# Train CIFAR10 CNN
python assignment2/train.py

# Train GAN
python assignment3/train.py
```

3. Start the APIs:
```bash
# Start Assignment 2 API (CIFAR10)
uvicorn assignment2.api:app --port 8000 --reload

# Start Assignment 3 API (GAN)
uvicorn assignment3.api:app --port 8001 --reload
```

## API Usage

### Assignment 2: CIFAR10 Image Classification

Send a POST request to `/predict` with an image file to get classification results:

```bash
curl -X POST -F "file=@your_image.jpg" http://localhost:8000/predict
```

### Assignment 3: GAN Digit Generation

Send a GET request to `/generate` to generate a handwritten digit:

```bash
# Generate a random digit
curl http://localhost:8001/generate

# Generate a specific digit (0-9)
curl "http://localhost:8001/generate?digit=5"
```

Response format:
```json
{
    "label": "predicted_class",
    "confidence": 0.9234
}
```
