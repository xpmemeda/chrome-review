#!/bin/bash

CPYTHON_VERSION=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --version) CPYTHON_VERSION="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done
if [ -z "$CPYTHON_VERSION" ]; then
    echo "usage: $0 --version <version>"
    exit 1
fi

INSTALL_DIR="$HOME/local/cpython-$CPYTHON_VERSION"
TEMP_DIR="$HOME/local/build.cpython-$CPYTHON_VERSION"

sudo yum groupinstall -y "Development Tools"
sudo yum install -y wget gmp-devel mpfr-devel libmpc-devel texinfo
sudo yum-builddep python3 -y

mkdir -p $TEMP_DIR
cd $TEMP_DIR

if [ ! -d "${HOME}/local/openssl-1.1.1w" ]; then
    wget https://www.openssl.org/source/openssl-1.1.1w.tar.gz
    tar -xzf openssl-1.1.1w.tar.gz
    cd openssl-1.1.1w
    ./config --prefix=$HOME/local/openssl-1.1.1w
    make -j && make install
fi

if [ ! -d "${INSTALL_DIR}" ]; then
    wget https://github.com/python/cpython/archive/refs/tags/v$CPYTHON_VERSION.zip
    if [ $? -ne 0 ]; then
        echo "[ERR]wget https://github.com/python/cpython/archive/refs/tags/v$CPYTHON_VERSION.zip"
        exit 1
    fi
    unzip v$CPYTHON_VERSION.zip && cd cpython-$CPYTHON_VERSION
    # Use --enable-shared to generate libpython{MAJOR}.{MINOR}.so. see detail: ./configure --help
    # Build both shared and static libraries.
    mkdir build-shared && cd build-shared && ../configure --prefix=$INSTALL_DIR --enable-shared --enable-optimizations --with-openssl=$HOME/local/openssl-1.1.1w && make -j && make install
    cd ..
    mkdir build-static && cd build-static && ../configure --prefix=$INSTALL_DIR --enable-optimizations --with-openssl=$HOME/local/openssl-1.1.1w && make -j && make install
    cd ${INSTALL_DIR}
    ln -s lib lib64
    cd ${INSTALL_DIR}/bin
    ln -s python3 python
fi

cd /
rm -rf $TEMP_DIR
