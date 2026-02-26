Copy

# ============================================================
# Production Face Validation Engine — Dockerfile
# Render.com compatible — no requirements.txt needed
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

# ── Python dependencies ───────────────────────────────────────
# Step 1: Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Step 2: Install PyTorch CPU (must be separate — uses custom index)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Step 3: Install all other packages
# Pillow==10.2.0  ← facenet-pytorch 2.6.0 requires Pillow>=10.2.0,<10.3.0
# numpy left unpinned ← mediapipe resolves its own numpy requirement
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
RUN mkdir -p /tmp/csv_jobs && chmod 777 /tmp/csv_jobs

# ── Non-root user ─────────────────────────────────────────────
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
