#!/bin/bash

GDB_VERSION="15.2"
INSTALL_DIR="$HOME/local/gdb-$GDB_VERSION"
TEMP_DIR="$HOME/local/build.$GDB_VERSION"

sudo yum groupinstall -y "Development Tools"
sudo yum install -y texinfo wget


mkdir -p $TEMP_DIR
cd $TEMP_DIR

wget "https://ftp.gnu.org/gnu/gdb/gdb-$GDB_VERSION.tar.gz"

tar -xzf gdb-$GDB_VERSION.tar.gz
cd gdb-$GDB_VERSION

mkdir build
cd build

../configure --prefix=$INSTALL_DIR

make -j$(nproc) && make install

cd $HOME/local
ln -s gdb-$GDB_VERSION gdb

cd /
rm -rf $TEMP_DIR

echo "GDB $GDB_VERSION has been installed in $INSTALL_DIR"
