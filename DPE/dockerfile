# 使用一个基础的 Python 镜像
FROM python:3.8-slim

# 安装系统级依赖
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    python3-dev \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制本地的 DPE-main 文件夹到容器内
COPY . /app

# 安装 PyTorch 和 torchvision（支持 CUDA 11.3）
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 安装 Python 依赖以及 torchmetrics
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install torchmetrics

# 暴露容器的端口
EXPOSE 5000
