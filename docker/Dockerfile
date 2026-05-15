# =====================================================================
# v4_hybridadmet Docker image
#
#  Base    : NVIDIA CUDA 12.4 + cuDNN runtime on Ubuntu 22.04
#  Size    : ~8 GB (CUDA + torch + PyG + JRE + RDKit + PaDEL + Uni-Mol2)
#  Build   : docker build -f docker/Dockerfile -t openadmet-v4:latest .
#  Run     : see docker-compose.yml or run_v4.sh
# =====================================================================

ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC

# ---- System deps (Python 3.11, Java for PaDEL, build tools) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        default-jre-headless \
        git curl wget ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# ---- Python deps in layers (cacheable) ----
# Core scientific stack first (changes rarely → top layer for cache hit)
RUN pip install --upgrade pip wheel setuptools \
    && pip install \
        numpy==1.26.4 pandas scipy scikit-learn tqdm rdkit sympy

# Classical ML stack
RUN pip install lightgbm xgboost catboost

# PyTorch + PyG stack (the big one — pick a torch version matched to CUDA 12.4)
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install torch-geometric \
    && pip install torch-scatter torch-sparse torch-cluster \
        -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

# Domain-specific ML
RUN pip install lightning chemprop tabpfn unimol_tools padelpy

# ---- Project code (mounted in at runtime — but we also COPY a fallback) ----
WORKDIR /workspace
COPY . /workspace

# ---- Pre-download foundation checkpoints at build time so the first run
#      doesn't stall on a slow Zenodo / HF download.
#      (best-effort; if network blocked at build time, skip)
RUN python -c "\
import os, urllib.request, pathlib; \
p = pathlib.Path.home() / '.chemprop'; p.mkdir(exist_ok=True); \
url = 'https://zenodo.org/records/15460715/files/chemeleon_mp.pt'; \
target = p / 'chemeleon_mp.pt'; \
target.exists() or urllib.request.urlretrieve(url, target); \
print('CheMeleon cached:', target)" || echo "CheMeleon prefetch skipped"

# ---- Entrypoint ----
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["v4"]
