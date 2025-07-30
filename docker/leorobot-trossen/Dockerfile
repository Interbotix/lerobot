# syntax=docker/dockerfile:1

FROM ubuntu:22.04

# Install base dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    bzip2 \
    libx11-6 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean


RUN apt-get install -y cmake build-essential  pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config
# Set environment variables
ENV MINICONDA_DIR=/root/miniconda3 \
    CONDA_ENV_NAME=lerobot

# Install Miniconda
RUN mkdir -p $MINICONDA_DIR && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $MINICONDA_DIR/miniconda.sh && \
    bash $MINICONDA_DIR/miniconda.sh -b -u -p $MINICONDA_DIR && \
    rm $MINICONDA_DIR/miniconda.sh

ENV PATH="$MINICONDA_DIR/bin:$PATH"

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# Create Conda environment and install ffmpeg
RUN conda create -y -n $CONDA_ENV_NAME python=3.10 && \
    conda install -y -n $CONDA_ENV_NAME ffmpeg -c conda-forge

# Clone LeRobot repository
RUN git clone https://github.com/huggingface/lerobot.git /lerobot
WORKDIR /lerobot

# Activate env and install LeRobot with [pi0] extra
RUN /bin/bash -c "source $MINICONDA_DIR/bin/activate $CONDA_ENV_NAME && \
    pip install -e ."

RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Use bash as default shell
SHELL ["/bin/bash", "-c"]

# Default command
CMD ["bash"]