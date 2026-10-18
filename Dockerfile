FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Build arg lets you pick the PyTorch wheel index:
#   CPU only (small, edge devices):  https://download.pytorch.org/whl/cpu
#   CUDA 12.x GPU:                   https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG INSTALL_EXTRAS=true

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt requirements-extra.txt ./
RUN pip install torch torchvision --index-url ${TORCH_INDEX} \
    && pip install -r requirements.txt \
    && if [ "$INSTALL_EXTRAS" = "true" ]; then pip install -r requirements-extra.txt; fi

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY supabase ./supabase
# Trained weights are built into the image (about 135 MB). With docker-compose, ./models is mounted over /srv/models
# instead, so you can swap models without rebuilding.
COPY models ./models
COPY yolov8n.pt ./yolov8n.pt

RUN useradd -m appuser && mkdir -p /srv/data && chown -R appuser /srv
USER appuser

# Render (and most hosts) tell the app which port to use through $PORT; locally it stays 8000.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
