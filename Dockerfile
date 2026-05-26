# =============================================================================
# flame3d-core Dockerfile
# Creates two conda environments: flame3d-core (project deps) and sam3 (SAM 3)
# Project code is volume-mounted at runtime — no rebuild needed for code changes.
# =============================================================================

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- System dependencies ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        git \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Miniconda ------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# Initialise conda for bash (so `conda activate` works in scripts)
RUN conda init bash

# Accept Anaconda channel Terms of Service (required in non-interactive builds)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ---- Create flame3d-core environment ----------------------------------------
RUN conda create -y -n flame3d-core python=3.12

# Copy only requirements.txt for layer caching — project code is volume-mounted
COPY requirements.txt /tmp/requirements.txt

# Install flame3d-core dependencies (includes Flask)
RUN conda run -n flame3d-core pip install --no-cache-dir \
        -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ---- Create sam3 environment ------------------------------------------------
RUN conda create -y -n sam3 python=3.12

# Install PyTorch with CUDA support in sam3 env
RUN conda run -n sam3 pip install --no-cache-dir \
        torch==2.10.0 torchvision \
        --index-url https://download.pytorch.org/whl/cu128

# Clone SAM 3 and install in editable mode
RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && conda run -n sam3 pip install --no-cache-dir -e /opt/sam3

# Optional: faster inference dependencies
RUN conda run -n sam3 pip install --no-cache-dir einops ninja \
    && conda run -n sam3 pip install --no-cache-dir flash-attn-3 --no-deps \
        --index-url https://download.pytorch.org/whl/cu128 || true

# ---- Working directory (will be overridden by volume mount) -----------------
WORKDIR /app

# ---- Default command: start the Flask server --------------------------------
# Uses conda run to activate the flame3d-core env, then launches Flask
EXPOSE 5005

CMD ["conda", "run", "--no-capture-output", "-n", "flame3d-core", \
     "python", "-m", "flask", "--app", "server:create_app()", "run", \
     "--host", "0.0.0.0", "--port", "5005"]
