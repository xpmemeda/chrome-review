#!/bin/bash

if [[ "$0" == "$BASH_SOURCE" ]]; then
    echo "Error: This script must be sourced, not executed."
    exit 1
fi

# cpython-3.12.12 can't install torch==2.1.2

source ${HOME}/envs/cpython-3.10.18/wnr/bin/activate
source ${HOME}/workspace/chrome-review/scripts/activate-dirs \
    -d /usr/local/cuda-12.4 \
    -d ${HOME}/local/gcc-x-7.5 \
    -d ${HOME}/local/llvm-18.1.5-gcc7.5-cxx03abi
