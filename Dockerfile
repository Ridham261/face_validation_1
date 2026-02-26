FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "Pillow==10.2.0" \
    "facenet-pytorch==2.6.0" \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "opencv-python-headless==4.9.0.80" \
    "mediapipe==0.10.14" \
    "numpy==1.26.4" \
    "aiohttp==3.9.5" \
    "pandas==2.2.2" \
    "python-multipart==0.0.9"

COPY main.py /app/main.py

RUN mkdir -p \
    /app/detection_outputs/front \
    /app/detection_outputs/side \
    /app/detection_outputs/mixed \
    /app/detection_outputs/eyes_closed \
    /app/detection_outputs/multiple_person \
    /app/detection_outputs/sunglasses \
    /app/detection_outputs/no_human \
    /app/detection_outputs/irrelevant \
    /app/detection_outputs/no_landmarks \
    /tmp/csv_jobs

RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app /tmp/csv_jobs \
    && chmod 777 /tmp/csv_jobs

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:8000/docs || exit 1

CMD ["python", "main.py"]
