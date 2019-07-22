# syntax=docker/dockerfile:experimental

# A multi-stage Dockerfile defining three separate images
# - compile-image: compile and build most of all dependencies
# - dev-image: for maskrcnn-benchmark development
# - app-image: deployable image

ARG CUDA="10.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04 as compile-image

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
        python3-virtualenv \
        python3-pip \
        python3-setuptools &&\
    apt-get autoremove --purge &&\
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY docker/inn/requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

WORKDIR /build

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
    && cd cocoapi/PythonAPI \
    && python3 setup.py bdist_wheel \
    && pip install dist/*

# install apex
RUN git clone https://github.com/NVIDIA/apex.git \
    && cd apex \
    && pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install IDF dependencies
RUN --mount=type=ssh mkdir -p -m 0600 ~/.ssh \
    && ssh-keyscan -p 7999 git.ba.innovatrics.net >> ~/.ssh/known_hosts \
    && pip wheel git+ssh://git@git.ba.innovatrics.net:7999/~pavel.svoboda/inm.git@1.0rc6#egg=inm --wheel-dir=/dist \
    && pip wheel git+ssh://git@git.ba.innovatrics.net:7999/~pavel.svoboda/idf.git@master#egg=idf --wheel-dir=/dist \
    && pip install --no-cache-dir /dist/*

WORKDIR /app
COPY . .

# Fool the image to have maskrcnn-benchmark installed in development
# mode in /opt/venv/lib/python3.6/site-packages referencing to /app
# The/app is thrown away and replaced by a bind of actual
# maskrcnn-benchmark source in the dev-image
RUN python setup.py bdist_wheel

# Development Image

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04 as dev-image

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

LABEL com.innovatrics.dev-maskrcnn="1.0.0"

RUN apt-get update && apt-get -y install --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3 \
        python3-dev \
        python3-distutils \
    && apt-get autoremove --purge \
    && rm -rf /var/lib/apt/lists/*

COPY --from=compile-image /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Deployment Image

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-runtime-ubuntu18.04 as app-image

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

LABEL com.innovatrics.dev-maskrcnn="1.0.0"

RUN apt-get update && apt-get -y install --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3 \
        python3-distutils \
    && apt-get autoremove --purge \
    && rm -rf /var/lib/apt/lists/*

COPY --from=compile-image /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --from=compile-image /app/dist/ /opt/dist/

RUN pip install --upgrade --no-cache-dir /opt/dist/* \
    && rm -rf /opt/dist

WORKDIR /app
