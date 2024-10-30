#!/bin/bash

version="4.1.6"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --version) version="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done
if [ -z "$version" ]; then
    echo "usage: $0 --version <version>"
    exit 1
fi
major_version=$(echo $version | cut -d. -f1)
minor_version=$(echo $version | cut -d. -f2)
INSTALL_DIR="$HOME/local/openmpi-${version}"
TEMP_DIR="$HOME/local/build.openmpi-${version}"

mkdir -p $TEMP_DIR
cd $TEMP_DIR

if [ ! -d ${INSTALL_DIR} ]; then
    wget https://download.open-mpi.org/release/open-mpi/v${major_version}.${minor_version}/openmpi-${version}.tar.gz
    if [ $? -ne 0 ]; then
        echo "[ERR]wget https://download.open-mpi.org/release/open-mpi/v${major_version}.${minor_version}/openmpi-${version}.tar.gz"
        exit 1
    fi
    tar -xvf openmpi-${version}.tar.gz
    cd openmpi-${version}
    ./configure --prefix=${INSTALL_DIR}
    make -j && make install
fi

cd /
rm -rf ${TEMP_DIR}
