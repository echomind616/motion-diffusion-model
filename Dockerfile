# Use a CUDA-enabled base image for MDM
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Istanbul

RUN apt-get update && apt-get install -y \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    blender \
    && rm -rf /var/lib/apt/lists/*

# Copy local files
COPY . .

# Install Python dependencies for MDM and Runpod
RUN pip install --no-cache-dir \
    runpod \
    tqdm \
    numpy \
    scipy \
    requests \
    python-multipart \
    httpx

# Install MDM specific libs
RUN pip install --no-cache-dir \
    spacy \
    joblib \
    matplotlib \
    ftfy \
    regex \
    gitpython \
    pytorch-lightning \
    clip \
    chumpy \
    smplx \
    trimesh \
    pyyaml \
    huggingface_hub

# Install CLIP from source if needed (MDM often requires specific versions)
RUN pip install git+https://github.com/openai/CLIP.git

# Command to run the handler
CMD [ "python", "-u", "runpod_handler.py" ]
