#!/bin/bash

GCC_VERSION="10.4.0"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --version) GCC_VERSION="$2"; shift ;;
        *) echo "unknown param: $1"; exit 1 ;;
    esac
    shift
done
INSTALL_DIR="$HOME/local/gcc-$GCC_VERSION"

if [ -d ${INSTALL_DIR} ]; then
    echo "gcc-${GCC_VERSION} had been installed."
    exit 0
fi

# Update system
sudo yum groupinstall -y "Development Tools"
sudo yum install -y wget gmp-devel mpfr-devel libmpc-devel texinfo

cache_path=/tmp/gcc-${GCC_VERSION}.tar.gz
if [ ! -f ${cache_path} ]; then
    wget https://ftp.gnu.org/gnu/gcc/gcc-$GCC_VERSION/gcc-$GCC_VERSION.tar.gz -O ${cache_path}
    if [ $? -ne 0 ]; then
        echo "download file err"
        exit 1
    fi
fi
tar -xzf ${cache_path} -C /tmp && cd /tmp/gcc-${GCC_VERSION}

# use sudo nethogs eth1 to watch downloading speed.
./contrib/download_prerequisites
# use --disable-multilib to build lib64 only.
mkdir build && cd build && ../configure --prefix=$INSTALL_DIR --enable-languages=c,c++ --disable-multilib && make -j$(nproc) && make install

cd / && rm -rf /tmp/gcc-${GCC_VERSION}
