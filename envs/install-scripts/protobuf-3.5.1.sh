#!/bin/bash

PROTOBUF_VERSION="3.5.1"
INSTALL_DIR="$HOME/local/protobuf-$PROTOBUF_VERSION"
TEMP_DIR="$HOME/local/build.$PROTOBUF_VERSION"

sudo yum groupinstall -y "Development Tools"
sudo yum install -y texinfo wget


mkdir -p $TEMP_DIR
cd $TEMP_DIR

wget https://github.com/protocolbuffers/protobuf/archive/refs/tags/v$PROTOBUF_VERSION.zip
unzip v$PROTOBUF_VERSION.zip
cd protobuf-$PROTOBUF_VERSION

./autogen.sh && ./configure --prefix=$INSTALL_DIR
make -j$(nproc) && make install

cd $HOME/local
ln -s protobuf-$PROTOBUF_VERSION protobuf

cd /
rm -rf $TEMP_DIR

echo "Protobuf $PROTOBUF_VERSION has been installed in $INSTALL_DIR"