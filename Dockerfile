# molvae designer — production image (GPU). For CPU-only, see the note at the bottom.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app/molvae

# Torch (CUDA 12.4) first so it layer-caches; then the lighter deps.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
 && pip install --no-cache-dir rdkit selfies numpy tqdm \
        "fastapi>=0.110" "uvicorn[standard]>=0.29" "openai>=1.40" "python-dotenv>=1.0"

# App code (artifacts are mounted at /artifacts, not baked in — keeps the image lean).
COPY . /app/molvae

ENV MOLVAE_ART_DIR=/artifacts \
    MOLVAE_DEVICE=cuda \
    PORT=8000
EXPOSE 8000

# 1 worker = 1 model copy in GPU memory. Scale with container replicas, not workers.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ---- CPU-only build: change the base image to `python:3.11-slim`, drop the
# `--index-url ...cu124` (use the default CPU torch wheel), and set MOLVAE_DEVICE=cpu.
