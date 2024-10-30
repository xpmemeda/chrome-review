#!/bin/bash

GCC_VERSION=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --version|-v) GCC_VERSION="$2"; shift ;;
        *) echo "unknown param: $1"; exit 1 ;;
    esac
    shift
done
if [ -z "$GCC_VERSION" ]; then
    echo "usage: $0 --version[-v] <version>"
    echo "example: $0 -v 10.5.0"
    exit 1
fi

INSTALL_DIR="$HOME/local/gcc-$GCC_VERSION"

if [ -d ${INSTALL_DIR} ]; then
    echo "gcc-${GCC_VERSION} had been installed."
    exit 0
fi

# Update system
sudo yum groupinstall -y "Development Tools"
sudo yum install -y wget gmp-devel mpfr-devel libmpc-devel texinfo

CACHE_DIR=$(pwd)
if [ ! -f ${CACHE_DIR}/gcc-$GCC_VERSION.tar.gz ]; then
    wget https://github.com/gcc-mirror/gcc/archive/refs/tags/releases/gcc-$GCC_VERSION.tar.gz
    if [ $? -ne 0 ]; then
        echo "download file err"
        exit 1
    fi
fi

tar -xzf gcc-$GCC_VERSION.tar.gz
if [[ -d gcc-$GCC_VERSION ]]; then
    cd gcc-$GCC_VERSION
elif [[ -d gcc-releases-gcc-$GCC_VERSION ]]; then
    cd gcc-releases-gcc-${GCC_VERSION}
else
    echo unknown dir, please check.
    exit 1
fi

# use sudo nethogs eth1 to watch downloading speed.
./contrib/download_prerequisites
# use --disable-multilib to build lib64 only.
mkdir build && cd build && ../configure --prefix=$INSTALL_DIR --enable-languages=c,c++ --disable-multilib && make -j$(nproc) && make install

cd .. && rm -rf gcc-releases-gcc-$GCC_VERSION gcc-${GCC_VERSION} gcc-$GCC_VERSION.tar.gz
