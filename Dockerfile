# ##################################################################################
# Setup Nvidia CUDA for Jetson Nano
# ##################################################################################
ARG V_OS_MAJOR=18
ARG V_OS_MINOR=04
ARG V_OS=${V_OS_MAJOR}.${V_OS_MINOR}
FROM --platform=linux/arm64/v8 nvcr.io/nvidia/l4t-base:r32.6.1 AS l4t-base
ARG TIMEZONE=Asia/Tokyo
# t194 = Jetson Xavier NX
# t210 = Jetson Nano
ARG V_SOC=t194
ARG V_CUDA_MAJOR=10
ARG V_CUDA_MINOR=2
ARG V_L4T_MAJOR=32
ARG V_L4T_MINOR=6
ENV V_CUDA=${V_CUDA_MAJOR}.${V_CUDA_MINOR}
ENV V_CUDA_DASH=${V_CUDA_MAJOR}-${V_CUDA_MINOR}
# ENV V_L4T=r${V_L4T_MAJOR}.${V_L4T_MINOR}
ENV V_L4T=r${V_L4T_MAJOR}.${V_L4T_MINOR}
# Expose environment variables everywhere
ENV CUDA=${V_CUDA_MAJOR}.${V_CUDA_MINOR}
# Accept default answers for everything
ENV DEBIAN_FRONTEND=noninteractive
# Fix CUDA info
ARG DPKG_STATUS
# Set timezone
ENV TZ=${TIMEZONE}
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

# 日本向けミラーで高速に同期を行う
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates
RUN sed -i.bak '/security/! s|http://ports.ubuntu.com/ubuntu-ports/|https://ftp.udx.icscoe.jp/Linux/ubuntu-ports/|' /etc/apt/sources.list

RUN echo "$DPKG_STATUS" >> /var/lib/dpkg/status \
    && echo "[Builder] Installing Prerequisites" \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common curl gnupg2 apt-utils \
    ninja-build git cmake libjpeg-dev libopenmpi-dev libomp-dev ccache\
    libopenblas-dev libblas-dev libeigen3-dev python3-pip

RUN echo "[Builder] Installing CUDA Repository" \
    && curl https://repo.download.nvidia.com/jetson/jetson-ota-public.asc > /etc/apt/trusted.gpg.d/jetson-ota-public.asc \
    && echo "deb https://repo.download.nvidia.com/jetson/common ${V_L4T} main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && echo "deb https://repo.download.nvidia.com/jetson/${V_SOC} ${V_L4T} main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && echo "[Builder] Installing CUDA System" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-libraries-${V_CUDA_DASH} \
    cuda-libraries-dev-${V_CUDA_DASH} \
    cuda-nvtx-${V_CUDA_DASH} \
    cuda-minimal-build-${V_CUDA_DASH} \
    cuda-license-${V_CUDA_DASH} \
    cuda-command-line-tools-${V_CUDA_DASH} \
    nvidia-cudnn* \
    libnvvpi1 vpi1-dev \
    && ln -s /usr/local/cuda-${V_CUDA} /usr/local/cuda \
    && rm -rf /var/lib/apt/lists/*
# ##################################################################################
# Create PyTorch Download Layer
# We do this seperately since else we need to keep rebuilding
# ##################################################################################
FROM alpine/git AS download
# Set timezone
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone
# Configuration Arguments
# https://github.com/pytorch/pytorch
ARG V_PYTORCH=v1.10.0
# https://github.com/pytorch/vision
ARG V_TORCHVISION=v0.11.0
# Accept default answers for everything
ENV DEBIAN_FRONTEND=noninteractive
# Clone Source
RUN git clone --recursive --branch ${V_PYTORCH} https://github.com/pytorch/pytorch
RUN git clone --recursive --branch ${V_TORCHVISION} https://github.com/pytorch/vision.git
# ##################################################################################
# Build PyTorch for Jetson (with CUDA)
# ##################################################################################
FROM l4t-base AS build
# Configuration Arguments
ARG V_PYTHON_MAJOR=3
ARG V_PYTHON_MINOR=8
ENV V_CLANG=8
ENV V_PYTHON=${V_PYTHON_MAJOR}.${V_PYTHON_MINOR}
# Accept default answers for everything
ENV DEBIAN_FRONTEND=noninteractive
# Download Common Software
RUN apt-get update \
    && apt-get install -y clang clang-${V_CLANG} build-essential bash ca-certificates git wget cmake curl software-properties-common ffmpeg libsm6 libxext6 libffi-dev libssl-dev xz-utils zlib1g-dev liblzma-dev \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${V_CLANG} 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${V_CLANG} 100 \
    && update-alternatives --set clang /usr/bin/clang-${V_CLANG} \
    && update-alternatives --set clang++ /usr/bin/clang++-${V_CLANG}

# Setting up Python
WORKDIR /install
RUN apt-get update \
    && apt-get install -y python${V_PYTHON} python${V_PYTHON}-dev python${V_PYTHON}-venv python${V_PYTHON_MAJOR}-tk;

# PyTorch - Build - Source Code Setup (copy repos from download to build)
# COPY ./pytorch /pytorch
COPY --from=download /git/pytorch /pytorch
COPY --from=download /git/vision /vision

WORKDIR /pytorch
# PyTorch - Build - Prerequisites
# Set clang as compiler
# clang supports the ARM NEON registers
# GNU GCC will give "no expression error"

COPY jetson-torch.patch /pytorch/
RUN patch -p1 < jetson-torch.patch || :

ARG CC=clang
ARG CXX=clang++
# Set path to ccache
ARG PATH=/usr/lib/ccache:$PATH
# Other arguments
ARG USE_CUDA=ON
ARG USE_CUDNN=ON
ARG BUILD_CAFFE2_OPS=0
ARG USE_FBGEMM=0
ARG USE_FAKELOWP=0
ARG BUILD_TEST=0
ARG USE_MKLDNN=0
ARG USE_NNPACK=0
ARG USE_XNNPACK=0
ARG USE_QNNPACK=0
ARG USE_PYTORCH_QNNPACK=0
ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2"
ARG USE_NCCL=0
ARG USE_SYSTEM_NCCL=0
ARG USE_OPENCV=0
ARG USE_DISTRIBUTED=0
# PyTorch Build
RUN cd /pytorch \
    && rm -rf build/CMakeCache.txt || : \
    && python3.8 -m pip install -U pip \
    && python3.8 -m pip install wheel mock cython pillow \
    && python3.8 -m pip install scikit-build \
    && python3.8 -m pip install setuptools==58.3.0 \
    && python3.8 -m pip install -r requirements.txt

RUN python3.8 setup.py bdist_wheel

# Install the PyTorch wheel
RUN apt-get install -y libswresample-dev libswscale-dev libavformat-dev libavcodec-dev libavutil-dev \
    && cd /pytorch/dist/ \
    && pip3 install `ls`

# Torchvision Build
WORKDIR /vision
COPY jetson-torchvision.patch /vision/
RUN patch -p1 < jetson-torchvision.patch || :
ARG FORCE_CUDA=1
RUN python3.8 setup.py bdist_wheel

# # ##################################################################################
# # Prepare Artifact
# # ##################################################################################
FROM scratch AS artifact
COPY --from=build /pytorch/dist/* /
COPY --from=build /vision/dist/* /
