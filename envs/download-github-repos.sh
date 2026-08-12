#!/bin/bash

if [ ! -d "$HOME/workspace/github/NVTX" ]; then
    git clone https://github.com/NVIDIA/NVTX.git $HOME/workspace/github/NVTX
fi

if [ ! -d "$HOME/workspace/github/cutlass" ]; then
    git clone https://github.com/NVIDIA/cutlass.git $HOME/workspace/github/cutlass
fi

if [ ! -d "$HOME/workspace/github/cpython" ]; then
    git clone https://github.com/python/cpython.git $HOME/workspace/github/cpython
fi

if [ ! -d "$HOME/workspace/github/googletest" ]; then
    git clone https://github.com/google/googletest.git $HOME/workspace/github/googletest
fi

if [ ! -d "$HOME/workspace/github/pybind11" ]; then
    git clone https://github.com/pybind/pybind11.git $HOME/workspace/github/pybind11
fi

if [ ! -d "$HOME/workspace/github/json" ]; then
    git clone https://github.com/nlohmann/json.git $HOME/workspace/github/json
fi

if [ ! -d "$HOME/workspace/github/libzmq" ]; then
    git clone https://github.com/zeromq/libzmq.git $HOME/workspace/github/libzmq
fi

if [ ! -d "$HOME/workspace/github/cppzmq" ]; then
    git clone https://github.com/zeromq/cppzmq.git $HOME/workspace/github/cppzmq
fi

if [ ! -d "$HOME/workspace/github/cpp-httplib" ]; then
    git clone https://github.com/yhirose/cpp-httplib.git $HOME/workspace/github/cpp-httplib
fi

if [ ! -d "$HOME/workspace/github/ucx" ]; then
    git clone https://github.com/openucx/ucx.git $HOME/workspace/github/ucx
fi

if [ ! -d "$HOME/workspace/github/Mooncake" ]; then
    git clone https://github.com/kvcache-ai/Mooncake.git $HOME/workspace/github/Mooncake
fi

if [ ! -d "$HOME/workspace/github/spdlog" ]; then
    git clone https://github.com/gabime/spdlog.git $HOME/workspace/github/spdlog
fi

if [ ! -d "$HOME/workspace/github/etcd-cpp-apiv3" ]; then
    git clone https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git $HOME/workspace/github/etcd-cpp-apiv3
fi

if [ ! -d "$HOME/workspace/github/TensorRT" ]; then
    git clone https://github.com/NVIDIA/TensorRT.git $HOME/workspace/github/TensorRT
fi

if [ ! -d "$HOME/workspace/github/onnxruntime" ]; then
    git clone https://github.com/microsoft/onnxruntime.git $HOME/workspace/github/onnxruntime
fi

if [ ! -d "$HOME/workspace/github/highfive" ]; then
    git clone https://github.com/highfive-devs/highfive.git $HOME/workspace/github/highfive
fi
