#!/bin/bash

if [[ "$0" == "$BASH_SOURCE" ]]; then
    echo "Error: This script must be sourced, not executed."
    exit 1
fi

# cpython-3.12.12 can't install torch==2.1.2

source ${HOME}/envs/cpython-3.10.18/wnr-zixiao/bin/activate

source ${HOME}/workspace/chrome-review/scripts/activate-dirs \
    -d /home/wnr/local/llvm-mlir-18x \
    -d /home/wnr/local/oneDNN

export SVML_HOME=/data/home/wnr/local/svml
export FARM_HASH_HOME=/data/home/wnr/local/farmhash
export HIGHWAY_HASH_HOME=/data/home/wnr/local/highwayhash
export MLAS_HOME=/data/home/wnr/local/mlas
export TNNC_HOME=/data/home/wnr/local/tnnc

# export LIBRARY_PATH=/data/home/wnr/local/farmhash:/data/home/wnr/local/highwayhash:/data/home/wnr/local/mlas:$LIBRARY_PATH
