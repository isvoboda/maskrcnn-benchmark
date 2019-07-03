ARG CUDA="10.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update && apt-get -y install --no-install-recommends \
        apt-utils git curl \
        ca-certificates \
        bzip2 \
        cmake \
        tree \
        htop \
        bmon \
        iotop \
        g++ \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ssh-client \
        python3-dev \
        python3-pip \
        python3-setuptools &&\
    apt-get autoremove --purge &&\
    rm -rf /var/lib/apt/lists/*

# Install all the freezed python deps
COPY docker/requirements.txt .
RUN python3 -m pip install --upgrade pip --no-cache-dir &&\
    python3 -m pip install pip-tools --no-cache-dir &&\
    python3 -m pip install -r requirements.txt --no-cache-dir

# # install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python3 setup.py build_ext \
 && python3 -m pip install -e .

# install apex
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex \
 && python3 -m pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}

WORKDIR /app
