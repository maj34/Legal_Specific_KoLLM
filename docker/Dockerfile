FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
# Install necessary packages and Chrome dependencies
RUN apt-get update
RUN apt-get install -y \
    git \
    curl \
    ssh \
    locales \
    unzip \
    vim \
    wget \
    tmux \
    htop \
    build-essential \
    fonts-nanum \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.8 \
    && apt-get install -y python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python requirements
## 1. check python and pip version
WORKDIR /workspace
RUN alias python=python3.8
RUN alias pip=pip3
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
RUN chmod -R 777 /workspace
CMD ["bash"]