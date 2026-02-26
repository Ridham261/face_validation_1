# ============================================================
# Production Face Validation Engine — Dockerfile
# ============================================================
# Stack: FastAPI + MTCNN (facenet-pytorch) + MediaPipe + OpenCV
#        + PyTorch (CPU) + aiohttp + pandas
# Python : 3.10-slim
# Port   : 8000
# ============================================================

FROM python:3.10-slim

# ── System dependencies ──────────────────────────────────────
# libgl1 + libglib2.0-0      → OpenCV (cv2)
# libsm6 + libxext6          → OpenCV display/threading
# libxrender-dev             → OpenCV rendering
# libgomp1                   → OpenMP (PyTorch CPU multi-thread)
# git                        → facenet-pytorch may pull GitHub weights
# wget + ca-certificates     → model weight downloads at runtime
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

# ── Python dependencies ──────────────────────────────────────
# Copy requirements first for Docker layer cache efficiency
COPY requirements.txt .

# Install PyTorch CPU-only first (smaller image, no CUDA bloat)
# Then install the rest
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

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
