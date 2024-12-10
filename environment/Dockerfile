# Use an existing image with Ubuntu, CUDA, and cuDNN
FROM registry.codeocean.com/codeocean/ubuntu:20.04-cuda11.7.0-cudnn8

ARG DEBIAN_FRONTEND=noninteractive

# Install necessary system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install torch and torchvision with CUDA support
RUN pip3 install --upgrade pip \
    && pip3 install torch torchvision 
RUN pip3 install scipy pandas scikit-image opencv-python

