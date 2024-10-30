FROM centos:centos7.9.2009 AS centos_base_builder

RUN yum update -y \
    && yum install wget bzip2 gcc make yum-utils which -y \
    && yum groupinstall "Development Tools" -y \
    && yum-builddep python3 -y
RUN mkdir /workspace

# Install cmake
FROM centos_base_builder AS cmake-3.26.3_builder
RUN cd /workspace \
    && wget https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3-linux-x86_64.sh \
    && chmod +x cmake-3.26.3-linux-x86_64.sh && echo "y\ny\n" | ./cmake-3.26.3-linux-x86_64.sh \
    && ln -s /workspace/cmake-3.26.3-linux-x86_64/bin/* /usr/local/bin

# Install binutils
FROM centos_base_builder AS binutils-2.41_builder
RUN cd /workspace \
    && wget https://ftp.gnu.org/gnu/binutils/binutils-2.41.tar.gz --no-check-certificate \
    && tar -xzf binutils-2.41.tar.gz \
    && cd binutils-2.41 \
    && ./configure --prefix=/usr/local/binutils-2.41 \
    && make -j5 \
    && make install \
    && rm -rf binutils-2.41.tar.gz binutils-2.41

# Install gcc-7.5.0
FROM centos_base_builder AS gcc-7.5.0_builder
RUN cd /workspace \
    && curl -O https://ftp.gnu.org/gnu/gcc/gcc-7.5.0/gcc-7.5.0.tar.gz \ 
    && tar -xzf gcc-7.5.0.tar.gz \
    && cd gcc-7.5.0 \
    && sed -i "s/ftp:\/\/gcc.gnu.org\/pub\/gcc\/infrastructure\//http:\/\/www.mirrorservice.org\/sites\/sourceware.org\/pub\/gcc\/infrastructure\//g" contrib/download_prerequisites \
    && ./contrib/download_prerequisites \
    && mkdir build-release \
    && cd build-release \
    && ../configure --prefix=/usr/local/gcc-7.5.0 --enable-languages=all --disable-multilib --enable-shared --enable-threads=posix --enable-__cxa_atexit --enable-clocale=gnu \
    && make -j5 \
    && make install \
    && rm -rf gcc-7.5.0.tar.gz gcc-7.5.0

FROM centos_base_builder AS base_builder
COPY --from=cmake-3.26.3_builder /workspace/cmake-3.26.3-linux-x86_64 /usr/local/cmake-3.26.3
COPY --from=binutils-2.41_builder /usr/local/binutils-2.41 /usr/local/binutils-2.41
COPY --from=gcc-7.5.0_builder /usr/local/gcc-7.5.0 /usr/local/gcc-7.5.0

ENV CC=/usr/local/gcc-7.5.0/bin/gcc 
ENV CXX=/usr/local/gcc-7.5.0/bin/g++
ENV PATH=/usr/local/cmake-3.26.3/bin:/usr/local/gcc-7.5.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/gcc-7.5.0/lib64

# Install farmhash master
FROM base_builder AS farmhash-x_builder
RUN cd /workspace && wget https://ftp.gnu.org/gnu/automake/automake-1.14.tar.gz
RUN cd /workspace && tar -xvf automake-1.14.tar.gz && cd automake-1.14 &&\
    mkdir build && cd build &&\
    ../configure --prefix=/usr/local/automake-1.14 &&\
    make -j && make install
ENV PATH=/usr/local/automake-1.14/bin:$PATH
RUN git clone https://github.com/google/farmhash.git &&\
    cd farmhash && mkdir build && cd build &&\
    ../configure --prefix=/usr/local/farmhash-x --with-pic &&\
    make -j && make install

# Install highwayhash master
FROM base_builder AS highwayhash-x_builder
RUN cd /workspace &&\
    git clone https://github.com/google/highwayhash.git &&\
    cd highwayhash &&\
    sed -i "s/usr\/local/usr\/local\/highwayhash-x/g" Makefile &&\
    make -j && make install

# Install oneDNN-1.7
FROM base_builder AS onednn-1.7_builder
RUN cd /workspace &&\
    wget https://github.com/oneapi-src/oneDNN/archive/refs/tags/v1.7.zip &&\
    unzip v1.7.zip &&\
    cd oneDNN-1.7 && mkdir build && cd build &&\
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/oneDNN-1.7 -DCMAKE_BUILD_TYPE=Release -DDNNL_LIBRARY_TYPE=STATIC &&\
    make -j && make install

# Install onnxruntime-1.15.1
FROM base_builder AS onnxruntime-1.15.1_builder
RUN cd /workspace &&\
    git clone https://github.com/microsoft/onnxruntime.git && cd onnxruntime && git checkout v1.15.1 &&\
    ./build.sh --config Release --parallel --allow_running_as_root --compile_no_warning_as_error --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=/usr/local/onnxruntime-1.15.1
RUN cd /workspace/onnxruntime/build/Linux/Release && make install
RUN mkdir /usr/local/mlas-1.15.1 &&\
    cp /workspace/onnxruntime/build/Linux/Release/libonnxruntime_mlas.a /usr/local/mlas-1.15.1 &&\
    cp /workspace/onnxruntime/build/Linux/Release/libonnxruntime_common.a /usr/local/mlas-1.15.1 &&\
    cp /workspace/onnxruntime/build/Linux/Release/_deps/google_nsync-build/libnsync_cpp.a /usr/local/mlas-1.15.1 &&\
    cp /workspace/onnxruntime/onnxruntime/core/mlas/inc/mlas.h /usr/local/mlas-1.15.1 &&\
    cp /workspace/onnxruntime/onnxruntime/core/mlas/inc/mlas_float16.h /usr/local/mlas-1.15.1

# Install llvm-18.1.8-wnr
FROM base_builder AS llvm-18.1.8_builder
RUN cd /workspace && wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-18.1.8.zip && unzip llvmorg-18.1.8.zip
WORKDIR /workspace/llvm-project-llvmorg-18.1.8
RUN cmake -B build llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local/llvm-18.1.8
RUN cmake --build build -j && cmake --install build

# Install cuda-12.3
FROM base_builder AS ctk_builder
RUN cd /workspace && wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
RUN cd /workspace && chmod +x cuda_12.3.2_545.23.08_linux.run && ./cuda_12.3.2_545.23.08_linux.run --silent --toolkit

# Install cudnn-9.2.1
FROM base_builder AS cudnn-9.2.1_builder
RUN cd /workspace &&\
    wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.2.1.18_cuda12-archive.tar.xz &&\
    tar -xvf cudnn-linux-x86_64-9.2.1.18_cuda12-archive.tar.xz

# Install python3.8
FROM base_builder AS cpython-3.8.16_builder
RUN cd /workspace &&\
    wget https://github.com/openssl/openssl/archive/refs/tags/OpenSSL_1_1_1w.tar.gz &&\
    tar -xvf OpenSSL_1_1_1w.tar.gz &&\
    cd openssl-OpenSSL_1_1_1w &&\
    mkdir build &&\
    cd build &&\
    ../config --prefix=/usr/local/openssl-1.1.1w &&\
    make -j &&\
    make install
RUN cd /workspace &&\
    wget https://github.com/python/cpython/archive/refs/tags/v3.8.16.zip &&\
    unzip v3.8.16.zip && cd cpython-3.8.16 &&\
    ./configure --prefix=/usr/local/python-3.8.16 --enable-optimizations --with-openssl=/usr/local/openssl-1.1.1w --with-openssl-rpath=auto &&\
    make -j && make install
RUN cd /usr/local/python-3.8.16 && ln -s lib lib64 && cd bin && ln -s python3 python
ENV PATH=/usr/local/python-3.8.16/bin:$PATH
RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade py urllib3==1.26.15
RUN python -m pip install 'build[virtualenv]' twine>=5.1.1 wheel auditwheel numpy==1.24.4 torch==2.1.2 torchtyping pytest 'pytest-xdist[psutil]'
RUN python -m pip install pybind11-stubgen gputil
RUN python -m pip install black==24.8.0 clang-format==18.1.8

# Install patchelf-0.18.0
FROM base_builder AS patchelf-0.18.0_builder
RUN cd /workspace && wget https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz
RUN cd /workspace && tar -xvf patchelf-0.18.0-x86_64.tar.gz
RUN mkdir /usr/local/patchelf-0.18.0 && cp -r /workspace/bin /usr/local/patchelf-0.18.0 && cp -r /workspace/share /usr/local/patchelf-0.18.0

# Install protobuf-3.5.1
FROM base_builder AS protobuf-3.5.1_builder
RUN cd /workspace && wget https://github.com/protocolbuffers/protobuf/archive/refs/tags/v3.5.1.zip
RUN cd /workspace && unzip v3.5.1.zip && cd protobuf-3.5.1 &&\
    ./autogen.sh && ./configure --prefix=/usr/local/protobuf-3.5.1 &&\
    make -j && make install

FROM base_builder AS vectorclass-x_builder
RUN cd /workspace && git clone https://github.com/vectorclass/version1.git && mv version1 /usr/local/vectorclass-x

FROM base_builder AS googletest-1.12.1_builder
RUN cd /workspace && wget https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
RUN cd /workspace && unzip release-1.12.1.zip && cd googletest-release-1.12.1 &&\
    cmake . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/googletest-1.12.1 &&\
    cmake --build build -j && cmake --install build

FROM base_builder AS builder
COPY --from=farmhash-x_builder /usr/local/automake-1.14 /usr/local/automake-1.14
COPY --from=farmhash-x_builder /usr/local/farmhash-x /usr/local/farmhash-x
COPY --from=highwayhash-x_builder /usr/local/highwayhash-x /usr/local/highwayhash-x
COPY --from=onednn-1.7_builder /usr/local/oneDNN-1.7 /usr/local/oneDNN-1.7
COPY --from=onnxruntime-1.15.1_builder /usr/local/onnxruntime-1.15.1 /usr/local/onnxruntime-1.15.1
COPY --from=onnxruntime-1.15.1_builder /usr/local/mlas-1.15.1 /usr/local/mlas-1.15.1
COPY --from=llvm-18.1.8_builder /usr/local/llvm-18.1.8 /usr/local/llvm-18.1.8
COPY --from=ctk_builder /usr/local/cuda-12.3 /usr/local/cuda-12.3
COPY --from=cudnn-9.2.1_builder /workspace/cudnn-linux-x86_64-9.2.1.18_cuda12-archive /usr/local/cudnn-9.2.1.18
COPY --from=cpython-3.8.16_builder /usr/local/python-3.8.16 /usr/local/python-3.8.16
COPY --from=patchelf-0.18.0_builder /usr/local/patchelf-0.18.0 /usr/local/patchelf-0.18.0
COPY --from=protobuf-3.5.1_builder /usr/local/protobuf-3.5.1 /usr/local/protobuf-3.5.1
COPY --from=vectorclass-x_builder /usr/local/vectorclass-x /usr/local/vectorclass-x
COPY --from=googletest-1.12.1_builder /usr/local/googletest-1.12.1 /usr/local/googletest-1.12.1

RUN ln -s /usr/local/cuda-12.3 /usr/local/cuda

ENV PATH=/usr/local/python-3.8.16/bin:/usr/local/cuda-12.3/bin:/usr/local/patchelf-0.18.0/bin:$PATH
ENV CMAKE_PREFIX_PATH=/usr/local/llvm-18.1.8:/usr/local/oneDNN-1.7
ENV CUDNN_HOME=/usr/local/cudnn-9.2.1.18
ENV VECTORCLASS_HOME=/usr/local/vectorclass-x
