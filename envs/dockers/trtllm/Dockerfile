FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

RUN apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs
RUN pip3 install tensorrt_llm==0.12.0 -U --extra-index-url https://pypi.nvidia.com

RUN apt-get install wget
RUN wget https://github.com/NVIDIA/TensorRT-LLM/archive/refs/tags/v0.12.0.zip
RUN apt-get install unzip
RUN unzip v0.12.0.zip
RUN pip install wonderwords

# docker build . -t olafxiong/trtllm-env

RUN apt update &&\
    apt install -y --no-install-recommends gnupg &&\
    echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list &&\
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub &&\
    apt update &&\
    apt install nsight-systems-cli
