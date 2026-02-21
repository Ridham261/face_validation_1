# ============================================================
# Production Face Validation Engine — Dockerfile
# ============================================================
FROM python:3.10-slim

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
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

# Expose the API port
EXPOSE 8000

# Start the application
CMD ["python", "main.py"]
