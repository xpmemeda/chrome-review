#!/bin/bash

cd $HOME/local

wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.2.1.18_cuda12-archive.tar.xz
tar -xvf cudnn-linux-x86_64-9.2.1.18_cuda12-archive.tar.xz

rm -rf cudnn-linux-x86_64-9.2.1.18_cuda12-archive.tar.xz

ln -s cudnn-linux-x86_64-9.2.1.18_cuda12-archive cudnn
