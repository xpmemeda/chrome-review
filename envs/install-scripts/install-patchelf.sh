#!/bin/bash

PATCHELF_VERSION="0.17.2"
INSTALL_DIR="$HOME/local/patchelf-$PATCHELF_VERSION"
TEMP_DIR="$HOME/local/build.patchelf-$PATCHELF_VERSION"

sudo yum groupinstall -y "Development Tools"
sudo yum install -y wget gmp-devel mpfr-devel libmpc-devel texinfo

mkdir -p $TEMP_DIR
cd $TEMP_DIR

wget https://github.com/NixOS/patchelf/releases/download/$PATCHELF_VERSION/patchelf-$PATCHELF_VERSION-x86_64.tar.gz
mkdir -p $INSTALL_DIR
tar -xvf patchelf-$PATCHELF_VERSION-x86_64.tar.gz -C $INSTALL_DIR

cd $HOME/local
ln -s patchelf-$PATCHELF_VERSION patchelf

cd /
rm -rf $TEMP_DIR

echo "PatchELF $PATCHELF_VERSION has been installed in $INSTALL_DIR"
