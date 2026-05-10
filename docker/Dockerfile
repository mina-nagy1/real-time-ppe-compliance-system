# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── System dependencies (OpenCV needs these) ──────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgstreamer1.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY configs/     ./configs/
COPY src/         ./src/
COPY api/         ./api/
COPY dashboard/   ./dashboard/
COPY models/      ./models/

# ── Create output directories ─────────────────────────────────────────────────
RUN mkdir -p outputs/logs outputs/metrics outputs/heatmaps outputs/videos

# ── Expose FastAPI port ───────────────────────────────────────────────────────
EXPOSE 8000

# ── Run ───────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
