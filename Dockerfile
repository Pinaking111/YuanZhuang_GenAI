FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

    RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install "uvicorn[standard]" fastapi python-multipart numpy pillow matplotlib

RUN mkdir -p /app/assignment2 /app/assignment3

COPY assignment2/__init__.py /app/assignment2/
COPY assignment2/model.py /app/assignment2/
COPY assignment2/api.py /app/assignment2/
COPY assignment2/train.py /app/assignment2/
COPY assignment2/cnn_cifar10.pt /app/assignment2/

COPY assignment3/__init__.py /app/assignment3/
COPY assignment3/model.py /app/assignment3/
COPY assignment3/api.py /app/assignment3/
COPY assignment3/train.py /app/assignment3/
COPY assignment3/gan_model.pt /app/assignment3/

ENV PYTHONPATH=/app

EXPOSE 8000 8001

CMD uvicorn assignment2.api:app --host 0.0.0.0 --port 8000 & \
    uvicorn assignment3.api:app --host 0.0.0.0 --port 8001