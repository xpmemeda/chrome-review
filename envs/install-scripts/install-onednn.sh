#!/bin/bash

if [ -z "${TMPDIR}" ]; then
  echo "TMPDIR is not set."
  exit 1
fi

VERSION="1.7"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --version) VERSION="$2"; shift ;;
        *) echo "unknown param: $1"; exit 1 ;;
    esac
    shift
done
INSTALL_DIR="$HOME/local/oneDNN-$VERSION"

if [ -d ${INSTALL_DIR} ]; then
    echo "oneDNN-${VERSION} had been installed."
    exit 0
fi

cache_path=${TMPDIR}/oneDNN-${VERSION}.zip
if [ ! -f ${cache_path} ]; then
    wget https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${VERSION}.zip -O ${cache_path}
    if [ $? -ne 0 ]; then
        echo "download file err"
        exit 1
    fi
fi
unzip ${cache_path} -d ${TMPDIR} && cd ${TMPDIR}/oneDNN-${VERSION}

cmake . -B build -DCMAKE_INSTALL_PREFIX=${HOME}/local/oneDNN-${VERSION} -DCMAKE_BUILD_TYPE=Release -DDNNL_LIBRARY_TYPE=STATIC
cmake --build build -j && cmake --install build

cd / && rm -rf ${TMPDIR}/oneDNN-${VERSION}
