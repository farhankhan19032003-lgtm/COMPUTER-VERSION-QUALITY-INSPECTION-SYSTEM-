FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Build arg lets you pick the PyTorch wheel index:
#   CPU only (small, edge devices):  https://download.pytorch.org/whl/cpu
#   CUDA 12.x GPU:                   https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG INSTALL_EXTRAS=true

# libgl1, libxcb1, libsm6, libxext6, libxrender1: ultralytics pulls in the full opencv-python next to
# opencv-python-headless, and the full build needs these system libraries to import cv2 (without them the app
# dies on start with "libxcb.so.1: cannot open shared object file").
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libxcb1 libsm6 libxext6 libxrender1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt requirements-extra.txt ./
RUN pip install torch torchvision --index-url ${TORCH_INDEX} \
    && pip install -r requirements.txt \
    && if [ "$INSTALL_EXTRAS" = "true" ]; then pip install -r requirements-extra.txt; fi

# Keep ONLY the headless OpenCV (no GUI libraries needed), then fail the build here, not at start-up, if cv2 or
# torch cannot be imported.
RUN pip uninstall -y opencv-python opencv-contrib-python || true \
    && pip install --force-reinstall --no-deps opencv-python-headless \
    && python -c "import cv2, torch, numpy; print('cv2', cv2.__version__, '| torch', torch.__version__)"

# ImageNet weights the app would otherwise download from the internet on the first request (ResNet18 for the anomaly
# backbone, ResNet50 for naming the product, about 145 MB). Baking them in means a slow or blocked download can neither
# delay the first photo nor silently switch auto-detect off. They sit outside /srv and are world-readable.
ENV TORCH_HOME=/opt/torch
RUN python -c "import torchvision as tv; tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1); tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT)" \
    && chmod -R a+rX /opt/torch \
    && ls -l /opt/torch/hub/checkpoints

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
