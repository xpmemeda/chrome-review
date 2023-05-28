FROM centos:centos7.9.2009 as base_builder

RUN yum update -y \
    && yum install wget bzip2 gcc make yum-utils -y \
    && yum groupinstall "Development Tools" -y \
    && yum-builddep python3 -y
RUN mkdir /workspace

# Install cmake
FROM base_builder as cmake-3.26.3_builder
RUN cd /workspace \
    && wget https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3-linux-x86_64.sh \
    && chmod +x cmake-3.26.3-linux-x86_64.sh && echo "y\ny\n" | ./cmake-3.26.3-linux-x86_64.sh \
    && ln -s /workspace/cmake-3.26.3-linux-x86_64/bin/* /usr/local/bin

# Install binutils
FROM base_builder as binutils-2.41_builder
RUN cd /workspace \
    && wget https://ftp.gnu.org/gnu/binutils/binutils-2.41.tar.gz --no-check-certificate \
    && tar -xzf binutils-2.41.tar.gz \
    && cd binutils-2.41 \
    && ./configure --prefix=/usr/local/binutils-2.41 \
    && make -j \
    && make install \
    && rm -rf binutils-2.41.tar.gz binutils-2.41

# Install gcc7.5
FROM base_builder as gcc-7.5.0_builder
RUN cd /workspace \
    && curl -O https://ftp.gnu.org/gnu/gcc/gcc-7.5.0/gcc-7.5.0.tar.gz \ 
    && tar -xzf gcc-7.5.0.tar.gz \
    && cd gcc-7.5.0 \
    && sed -i "s/ftp:\/\/gcc.gnu.org\/pub\/gcc\/infrastructure\//http:\/\/www.mirrorservice.org\/sites\/sourceware.org\/pub\/gcc\/infrastructure\//g" contrib/download_prerequisites \
    && ./contrib/download_prerequisites \
    && mkdir build-release \
    && cd build-release \
    && ../configure --prefix=/usr/local/gcc-7.5.0 --enable-languages=all --disable-multilib --enable-shared --enable-threads=posix --enable-__cxa_atexit --enable-clocale=gnu \
    && make -j \
    && make install \
    && rm -rf gcc-7.5.0.tar.gz gcc-7.5.0

FROM base_builder as base_builder
COPY --from=cmake-3.26.3_builder /workspace/cmake-3.26.3-linux-x86_64 /usr/local/cmake-3.26.3-linux-x86_64
COPY --from=binutils-2.41_builder /usr/local/binutils-2.41 /usr/local/binutils-2.41
COPY --from=gcc-7.5.0_builder /usr/local/gcc-7.5.0 /usr/local/gcc-7.5.0

# Install python3.8
FROM base_builder as cpython-3.8.16_builder
ENV CC=/usr/local/gcc-7.5.0/bin/gcc \ 
    CXX=/usr/local/gcc-7.5.0/bin/g++ \
    LD_LIBRARY_PATH=/usr/local/gcc-7.5.0/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cmake-3.26.3-linux-x86_64/bin:$PATH
RUN cd /workspace \
    && wget https://github.com/python/cpython/archive/refs/tags/v3.8.16.zip \
    && unzip v3.8.16.zip \
    && cd cpython-3.8.16 \
    && ./configure --prefix=/usr/local/python-3.8.16 \
    && make -j \
    && make install

FROM base_builder as cpython-3.6.15_builder
ENV CC=/usr/local/gcc-7.5.0/bin/gcc \
    CXX=/usr/local/gcc-7.5.0/bin/g++ \
    LD_LIBRARY_PATH=/usr/local/gcc-7.5.0/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cmake-3.26.3-linux-x86_64/bin:$PATH
RUN cd /workspace \
    && wget https://github.com/python/cpython/archive/refs/tags/v3.6.15.zip \
    && unzip v3.6.15.zip \
    && cd cpython-3.6.15 \
    && ./configure --prefix=/usr/local/python-3.6.15 \
    && make -j \
    && make install

# Install llvm-16.0.6
FROM base_builder as llvm-16.0.6_builder
ENV CC=/usr/local/gcc-7.5.0/bin/gcc \ 
    CXX=/usr/local/gcc-7.5.0/bin/g++ \
    LD_LIBRARY_PATH=/usr/local/gcc-7.5.0/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cmake-3.26.3-linux-x86_64/bin:$PATH
RUN cd /workspace \
    && wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-16.0.6.zip \
    && unzip llvmorg-16.0.6.zip \
    && cd llvm-project-llvmorg-16.0.6 \
    && mkdir build \
    && cd build \
    && cmake ../llvm -DCMAKE_INSTALL_PREFIX=/usr/local/llvm-16.0.6 -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=ON \
    && make -j \
    && make install

# Install oneDNN-1.7
FROM base_builder as onednn-1.7_builder
ENV CC=/usr/local/gcc-7.5.0/bin/gcc \ 
    CXX=/usr/local/gcc-7.5.0/bin/g++ \
    LD_LIBRARY_PATH=/usr/local/gcc-7.5.0/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cmake-3.26.3-linux-x86_64/bin:$PATH
RUN cd /workspace \
    && wget https://github.com/oneapi-src/oneDNN/archive/refs/tags/v1.7.zip \
    && unzip v1.7.zip \
    && cd oneDNN-1.7 \
    && mkdir build \
    && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/oneDNN-1.7 -DCMAKE_BUILD_TYPE=Release -DDNNL_LIBRARY_TYPE=STATIC \
    && make -j \
    && make install

# Install CTK-11.7: https://developer.nvidia.com/cuda-toolkit-archive
FROM base_builder as ctk-11.7_builder
RUN yum install which -y
RUN cd /workspace \
    && wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run \
    && chmod +x cuda_11.7.1_515.65.01_linux.run \
    && ./cuda_11.7.1_515.65.01_linux.run --silent --toolkit

# Install protobuf-3.5.1
FROM base_builder as protobuf-3.5.1_builder
ENV CC=/usr/local/gcc-7.5.0/bin/gcc \ 
    CXX=/usr/local/gcc-7.5.0/bin/g++ \
    LD_LIBRARY_PATH=/usr/local/gcc-7.5.0/lib64:$LD_LIBRARY_PATH \
    PATH=/usr/local/gcc-7.5.0/bin:$PATH
RUN cd /workspace \
    && wget https://github.com/protocolbuffers/protobuf/archive/refs/tags/v3.5.1.zip \
    && unzip v3.5.1.zip \
    && cd protobuf-3.5.1 \
    && ./autogen.sh \
    && ./configure --prefix=/usr/local/protobuf-3.5.1 \
    && make -j \
    && make install

# Install onnxruntime-1.15.1
FROM base_builder as onnxruntime-1.15.1_builder
WORKDIR /workspace
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-gpu-1.15.1.tgz
RUN tar -xzf onnxruntime-linux-x64-gpu-1.15.1.tgz

# Install cudnn from host file. https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
FROM base_builder as cudnn-8.9.7.29-ctk-11_builder
WORKDIR /workspace
COPY cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz /workspace
RUN tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz

# Install gcc10.5
FROM base_builder as gcc-10.5.0_builder
RUN cd /workspace \
    && curl -O https://ftp.gnu.org/gnu/gcc/gcc-10.5.0/gcc-10.5.0.tar.gz \ 
    && tar -xzf gcc-10.5.0.tar.gz \
    && cd gcc-10.5.0 \
    && sed -i "s/ftp:\/\/gcc.gnu.org\/pub\/gcc\/infrastructure\//http:\/\/www.mirrorservice.org\/sites\/sourceware.org\/pub\/gcc\/infrastructure\//g" contrib/download_prerequisites \
    && ./contrib/download_prerequisites \
    && mkdir build-release \
    && cd build-release \
    && ../configure --prefix=/usr/local/gcc-10.5.0 --enable-languages=all --disable-multilib --enable-shared --enable-threads=posix --enable-__cxa_atexit --enable-clocale=gnu \
    && make -j \
    && make install \
    && rm -rf gcc-10.5.0.tar.gz gcc-10.5.0

FROM base_builder as base_builder
COPY --from=cpython-3.8.16_builder /usr/local/python-3.8.16 /usr/local/python-3.8.16
COPY --from=llvm-16.0.6_builder /usr/local/llvm-16.0.6 /usr/local/llvm-16.0.6
COPY --from=onednn-1.7_builder /usr/local/oneDNN-1.7 /usr/local/oneDNN-1.7
COPY --from=ctk-11.7_builder /usr/local/cuda-11.7 /usr/local/cuda-11.7
COPY --from=protobuf-3.5.1_builder /usr/local/protobuf-3.5.1 /usr/local/protobuf-3.5.1
COPY --from=onnxruntime-1.15.1_builder /workspace/onnxruntime-linux-x64-gpu-1.15.1 /usr/local/onnxruntime-linux-x64-gpu-1.15.1
COPY --from=cudnn-8.9.7.29-ctk-11_builder /workspace/cudnn-linux-x86_64-8.9.7.29_cuda11-archive /usr/local/cudnn-linux-x86_64-8.9.7.29_cuda11-archive
COPY --from=gcc-10.5.0_builder /usr/local/gcc-10.5.0 /usr/local/gcc-10.5.0
COPY --from=cpython-3.6.15_builder /usr/local/python-3.6.15 /usr/local/python-3.6.15

FROM base_builder as python3.8_libs_builder
RUN /usr/local/python-3.8.16/bin/python3.8 -m venv /workspace/python-envs/38
RUN source /workspace/python-envs/38/bin/activate && python -m pip install tensorflow urllib3==1.26.6
RUN source /workspace/python-envs/38/bin/activate && python -m pip install torch==2.0.0

FROM base_builder as python3.6_libs_builder
RUN /usr/local/python-3.6.15/bin/python3.6 -m venv /workspace/python-envs/36
RUN source /workspace/python-envs/36/bin/activate && python -m pip install urllib3==1.26.6

FROM base_builder as builder
COPY --from=python3.8_libs_builder /workspace/python-envs/38 /workspace/python-envs/38
COPY --from=python3.6_libs_builder /workspace/python-envs/36 /workspace/python-envs/36

# ENV.
RUN yum install which -y
COPY gcc-7.5.env gcc-10.5.env /workspace
WORKDIR /workspace
