# Use an official NVIDIA CUDA base image for PyTorch acceleration
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY advertest/requirements.txt /app/requirements.txt

# Install core dependencies and MLOps tools
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
    torchmetrics \
    pycocotools \
    label-studio-sdk \
    python-dotenv \
    wandb

# Clone YOLOv7-face into third_party (as required by inference_engine)
RUN mkdir -p /app/third_party && \
    git clone https://github.com/derronqi/yolov7-face.git /app/third_party/yolov7_face

# Copy the rest of the application code
COPY . /app/

# Set the default command to show CLI help
CMD ["python", "main.py", "--help"]
