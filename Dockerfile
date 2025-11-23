FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

    RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install "uvicorn[standard]" fastapi python-multipart numpy pillow matplotlib

RUN mkdir -p /app/assignment4

# Copy only assignment4 (latest homework)
COPY assignment4/ /app/assignment4/

ENV PYTHONPATH=/app

# Expose single port for grader convenience
EXPOSE 8000

# Start the Assignment 4 API
CMD ["uvicorn", "assignment4.api:app", "--host", "0.0.0.0", "--port", "8000"]