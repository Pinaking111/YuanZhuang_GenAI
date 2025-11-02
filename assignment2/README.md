# Assignment 2: CIFAR10 Image Classification

This directory contains a CNN implementation for CIFAR10 image classification, along with a FastAPI service for serving the model.

## API Endpoints

The API service runs on port 8000.

You can access the API documentation at: http://localhost:8000/docs

### Endpoints

1. `/predict` (POST)
   - Classify an uploaded image into one of the CIFAR10 categories
   - Parameters:
     - `file`: Image file to classify

Example usage:
```bash
curl -X POST -F "file=@your_image.jpg" http://localhost:8000/predict
```

## Local Development

To run the service locally:

```bash
uvicorn assignment2.api:app --port 8000 --reload
```

## Docker

When running in Docker (as configured in the root Dockerfile), the service automatically starts on port 8000.