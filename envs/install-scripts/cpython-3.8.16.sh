#!/bin/bash

CPYTHON_VERSION="3.8.16"
INSTALL_DIR="$HOME/local/cpython-$CPYTHON_VERSION"
TEMP_DIR="$HOME/local/build.cpython-$CPYTHON_VERSION"

sudo yum groupinstall -y "Development Tools"
sudo yum install -y wget gmp-devel mpfr-devel libmpc-devel texinfo
sudo yum-builddep python3 -y

mkdir -p $TEMP_DIR
cd $TEMP_DIR

# Install ssl.
wget https://www.openssl.org/source/openssl-1.1.1w.tar.gz
tar -xzf openssl-1.1.1w.tar.gz
cd openssl-1.1.1w

./config --prefix=$HOME/local/openssl-1.1.1w
make -j && make install

# Install cpython.
wget https://github.com/python/cpython/archive/refs/tags/v$CPYTHON_VERSION.zip
unzip v$CPYTHON_VERSION.zip && cd cpython-$CPYTHON_VERSION

./configure --prefix=$INSTALL_DIR --with-openssl=$HOME/local/openssl-1.1.1w --with-openssl-rpath=auto
make -j && make install

cd $HOME/local
ln -s cpython-$CPYTHON_VERSION cpython

cd $HOME/local/cpython/bin
ln -s python3 python

cd $HOME/local/cpython
ln -s lib lib64

cd /
rm -rf $TEMP_DIR

echo "Cpython $CPYTHON_VERSION has been installed in $INSTALL_DIR"
