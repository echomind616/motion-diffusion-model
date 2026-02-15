# Use a CUDA-enabled base image for MDM
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    tzdata \
    wget \
    ffmpeg \
    git \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    blender \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy local files
COPY . .

# Upgrade build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core/runtime dependencies
RUN pip install --no-cache-dir \
    runpod \
    tqdm \
    numpy \
    scipy \
    requests \
    python-multipart \
    httpx

# Install MDM standard dependencies
RUN pip install --no-cache-dir \
    spacy \
    joblib \
    matplotlib \
    ftfy \
    regex \
    gitpython \
    pytorch-lightning \
    trimesh \
    pyyaml \
    huggingface_hub

# Install MDM complex dependencies (often requires compilation/extras)
RUN pip install --no-cache-dir \
    chumpy \
    smplx

# Install CLIP from source
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Command to run the handler
CMD [ "python", "-u", "runpod_handler.py" ]
