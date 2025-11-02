# Assignment 3: GAN for Handwritten Digit Generation

This directory contains a GAN (Generative Adversarial Network) implementation for generating handwritten digits, along with a FastAPI service for serving the model.

## API Endpoints

The API service runs on port 8001 (to avoid conflict with Assignment 2's service on port 8000).

You can access the API documentation at: http://localhost:8001/docs

### Endpoints

1. `/generate` (GET)
   - Generate a random handwritten digit
   - Query Parameters:
     - `digit` (optional, int, 0-9): Generate a specific digit

Example usage:
```bash
# Generate a random digit
curl http://localhost:8001/generate

# Generate a specific digit (e.g., 5)
curl "http://localhost:8001/generate?digit=5"
```

## Local Development

To run the service locally:

```bash
uvicorn assignment3.api:app --port 8001 --reload
```

## Docker

When running in Docker (as configured in the root Dockerfile), the service automatically starts on port 8001.