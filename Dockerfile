FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install "uvicorn[standard]" fastapi python-multipart numpy pillow

# 创建 Python 包目录结构
RUN mkdir -p /app/assignment2

# 复制代码和模型
COPY assignment2/__init__.py /app/assignment2/
COPY assignment2/model.py /app/assignment2/
COPY assignment2/api.py /app/assignment2/
COPY assignment2/train.py /app/assignment2/
COPY assignment2/cnn_cifar10.pt /app/assignment2/

# 设置Python路径
ENV PYTHONPATH=/app

# 暴露API端口
EXPOSE 8000

# 设置启动命令
CMD ["uvicorn", "assignment2.api:app", "--host", "0.0.0.0", "--port", "8000"]