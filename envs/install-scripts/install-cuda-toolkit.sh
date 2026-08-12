#!/bin/bash

set -e

CUDA_VERSION="12.8.0"
CUDA_BUILD_VERSION="570.86.10"
CREATE_SYMLINK=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --version|-v) CUDA_VERSION="$2"; shift ;;
        --build-version|-b) CUDA_BUILD_VERSION="$2"; shift ;;
        --create-symlink) CREATE_SYMLINK=1 ;;
        *)
            echo "unknown param: $1"
            exit 1
            ;;
    esac
    shift
done

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[ERR]install-cuda-toolkit.sh only supports Linux"
    exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "[ERR]install-cuda-toolkit.sh only supports x86_64"
    exit 1
fi

CUDA_MAJOR_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f1-2)
INSTALL_DIR="/usr/local/cuda-${CUDA_MAJOR_MINOR}"
INSTALLER_NAME="cuda_${CUDA_VERSION}_${CUDA_BUILD_VERSION}_linux.run"
DOWNLOAD_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${INSTALLER_NAME}"
DOWNLOAD_DIR="${TMPDIR:-/tmp}"
INSTALLER_PATH="${DOWNLOAD_DIR}/${INSTALLER_NAME}"

if [ -d "$INSTALL_DIR" ]; then
    echo "[INFO]${INSTALL_DIR} already exists"
    exit 0
fi

if [ ! -f "$INSTALLER_PATH" ]; then
    echo "[INFO]download ${DOWNLOAD_URL}"
    if ! wget -O "$INSTALLER_PATH" "$DOWNLOAD_URL"; then
        echo "[ERR]wget ${DOWNLOAD_URL}"
        rm -f "$INSTALLER_PATH"
        exit 1
    fi
else
    echo "[INFO]reuse downloaded file: $INSTALLER_PATH"
fi

chmod +x "$INSTALLER_PATH"
sudo "$INSTALLER_PATH" --silent --toolkit

if [ "$CREATE_SYMLINK" -eq 1 ]; then
    sudo ln -sfn "$INSTALL_DIR" /usr/local/cuda
fi

echo "[INFO]installed CUDA toolkit to ${INSTALL_DIR}"
