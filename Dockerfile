FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (torch CPU wheel plus common libs and transformers stack)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "uvicorn[standard]" fastapi python-multipart numpy pillow matplotlib && \
    pip install --no-cache-dir transformers datasets accelerate sentencepiece

# Only copy assignment5 and the root main app to keep image small and focused
COPY main.py /app/main.py
COPY assignment5/ /app/assignment5/

ENV PYTHONPATH=/app

# Expose port for the FastAPI app
EXPOSE 8000

# Start the FastAPI app (root `main.py` serves assignment5)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]