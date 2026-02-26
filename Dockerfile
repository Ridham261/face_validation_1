# ============================================================
# Production Face Validation Engine — Dockerfile
# ============================================================
# Stack: FastAPI + MTCNN (facenet-pytorch) + MediaPipe + OpenCV
#        + PyTorch (CPU) + aiohttp + pandas
# Python : 3.10-slim  |  Port: 8000
# No requirements.txt needed — all deps embedded here.
# ============================================================

FROM python:3.10-slim

# ── System dependencies ──────────────────────────────────────
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

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (no requirements.txt needed) ─────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        fastapi==0.111.0 \
        "uvicorn[standard]==0.29.0" \
        facenet-pytorch==2.6.0 \
        opencv-python-headless==4.9.0.80 \
        mediapipe==0.10.14 \
        Pillow==10.3.0 \
        numpy==1.26.4 \
        aiohttp==3.9.5 \
        pandas==2.2.2 \
        python-multipart==0.0.9

# ── Application code ─────────────────────────────────────────
COPY main.py .

# ── Output directories ───────────────────────────────────────
RUN mkdir -p \
    detection_outputs/front \
    detection_outputs/side \
    detection_outputs/mixed \
    detection_outputs/eyes_closed \
    detection_outputs/multiple_person \
    detection_outputs/sunglasses \
    detection_outputs/no_human \
    detection_outputs/irrelevant \
    detection_outputs/no_landmarks

# ── Temp directory for CSV job outputs ───────────────────────
RUN mkdir -p /tmp/csv_jobs

# ── Non-root user (security best practice) ───────────────────
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /tmp/csv_jobs
USER appuser

# ── Port ─────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:8000/docs || exit 1

# ── Start ────────────────────────────────────────────────────
CMD ["python", "main.py"]
