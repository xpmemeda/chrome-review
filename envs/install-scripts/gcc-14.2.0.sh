#!/bin/bash

# Set gcc version and install dir.
GCC_VERSION="14.2.0"
INSTALL_DIR="$HOME/local/gcc-$GCC_VERSION"
TEMP_DIR="$HOME/local/build.gcc-$GCC_VERSION"

# Update system
sudo yum groupinstall -y "Development Tools"
sudo yum install -y wget gmp-devel mpfr-devel libmpc-devel texinfo

# Install GCC
mkdir -p $TEMP_DIR && cd $TEMP_DIR

GCC_URL="https://ftp.gnu.org/gnu/gcc/gcc-$GCC_VERSION/gcc-$GCC_VERSION.tar.gz"
wget $GCC_URL

tar -xzf gcc-$GCC_VERSION.tar.gz
cd gcc-$GCC_VERSION

./contrib/download_prerequisites

mkdir build
cd build

../configure --prefix=$INSTALL_DIR --enable-languages=c,c++ --disable-multilib

make -j$(nproc) && make install

cd $HOME/local
ln -s gcc-$GCC_VERSION gcc

cd /
rm -rf $TEMP_DIR

echo "GCC $GCC_VERSION has been installed in $INSTALL_DIR"
