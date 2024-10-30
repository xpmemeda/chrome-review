#!/bin/bash

VERSION=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --version|-v) VERSION="$2"; shift ;;
        *) echo "unknown params: $1"; exit 1 ;;
    esac
    shift
done
if [ -z "$VERSION" ]; then
    echo "usage: $0 --version[-v] <version>"
    echo "example: $0 -v 2.28"
    exit 1
fi

INSTALL_DIR="$HOME/local/glibc-$VERSION"

if [ -d ${INSTALL_DIR} ]; then
    echo "glibc-$VERSION had been installed."
    exit 0
fi

CACHE_DIR=$(pwd)
if [ ! -f ${CACHE_DIR}/glibc-$VERSION.tar.gz ]; then
    wget https://github.com/bminor/glibc/archive/refs/tags/glibc-$VERSION.tar.gz
    if [ $? -ne 0 ]; then
        echo "download file err"
        exit 1
    fi
fi

tar -xzf glibc-$VERSION.tar.gz
if [[ -d glibc-$VERSION ]]; then
    cd glibc-$VERSION
elif [[ -d glibc-glibc-$VERSION ]]; then
    cd glibc-glibc-$VERSION
else
    echo unknown dir, please check.
    exit 1
fi

mkdir build && cd build
../configure --prefix=$INSTALL_DIR
make -j
