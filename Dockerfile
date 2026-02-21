# ============================================================
# Production Face Validation Engine — Optimized Dockerfile
# ============================================================

FROM python:3.10-slim

# Prevent python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install minimal system dependencies for OpenCV & MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch (VERY IMPORTANT for Render)
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    torchvision==0.18.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    python-multipart==0.0.9 \
    opencv-python-headless==4.9.0.80 \
    "numpy>=1.24.0,<2.0.0" \
    "Pillow>=10.2.0,<10.3.0" \
    mediapipe==0.10.14 \
    requests \
    facenet-pytorch==2.6.0

# Copy application
COPY main.py .

# Create output directories
RUN mkdir -p detection_outputs/front \
             detection_outputs/side \
             detection_outputs/mixed \
             detection_outputs/eyes_closed \
             detection_outputs/multiple_person \
             detection_outputs/sunglasses \
             detection_outputs/no_human \
             detection_outputs/irrelevant

EXPOSE 8000

# Use uvicorn directly (better for production)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
