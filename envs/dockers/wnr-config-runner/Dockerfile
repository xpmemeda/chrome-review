FROM mirrors.tencent.com/tlinux/tlinux4.0 AS backbone_builder_stage_0

RUN yum update -y \
    && yum install wget bzip2 gcc make which gmp-devel mpfr-devel libmpc-devel texinfo -y \
    && yum groupinstall "Development Tools" -y \
    && yum-builddep python3 -y \
    && yum install git-lfs -y

RUN mkdir /.docker_workspace

ENV _docker_cmake_version="3.26.3"
FROM backbone_builder_stage_0 AS cmake_builder
RUN cd /.docker_workspace \
    && wget https://github.com/Kitware/CMake/releases/download/v${_docker_cmake_version}/cmake-${_docker_cmake_version}-linux-x86_64.sh \
    && chmod +x cmake-${_docker_cmake_version}-linux-x86_64.sh && echo "y\ny\n" | ./cmake-${_docker_cmake_version}-linux-x86_64.sh \
    && ln -s /.docker_workspace/cmake-${_docker_cmake_version}-linux-x86_64/bin/* /usr/local/bin

FROM backbone_builder_stage_0 AS backbone_builder_stage_1
COPY --from=cmake_builder /.docker_workspace/cmake-${_docker_cmake_version}-linux-x86_64 /usr/local/cmake-${_docker_cmake_version}

ENV CC=/usr/bin/gcc 
ENV CXX=/usr/bin/g++
ENV PATH=/usr/local/cmake-${_docker_cmake_version}/bin:$PATH

ENV _docker_cpython_version="3.10.15"
FROM backbone_builder_stage_1 AS cpython_builder
RUN cd /.docker_workspace &&\
    wget https://github.com/openssl/openssl/archive/refs/tags/OpenSSL_1_1_1w.tar.gz &&\
    tar -xvf OpenSSL_1_1_1w.tar.gz &&\
    cd openssl-OpenSSL_1_1_1w &&\
    mkdir build &&\
    cd build &&\
    ../config --prefix=/usr/local/openssl-1.1.1w &&\
    make -j &&\
    make install
RUN cd /.docker_workspace &&\
    wget https://github.com/python/cpython/archive/refs/tags/v${_docker_cpython_version}.zip &&\
    unzip v${_docker_cpython_version}.zip && cd cpython-${_docker_cpython_version} &&\
    ./configure --prefix=/usr/local/cpython-${_docker_cpython_version} --enable-optimizations --with-openssl=/usr/local/openssl-1.1.1w --with-openssl-rpath=auto &&\
    make -j && make install
RUN cd /usr/local/cpython-${_docker_cpython_version} && ln -s lib lib64 && cd bin && ln -s python3 python
ENV PATH=/usr/local/cpython-${_docker_cpython_version}/bin:$PATH
RUN python -m pip install --upgrade pip
RUN python -m pip install build twine wheel
RUN python -m pip install wnr==1.4.1rc12 --extra-index-url=https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple

FROM backbone_builder_stage_1 AS ctk_builder
RUN cd /.docker_workspace && wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
RUN cd /.docker_workspace && chmod +x cuda_12.3.2_545.23.08_linux.run && ./cuda_12.3.2_545.23.08_linux.run --silent --toolkit

FROM backbone_builder_stage_1 AS backbone_builder_stage_2
COPY --from=cpython_builder /usr/local/openssl-1.1.1w /usr/local/openssl-1.1.1w
COPY --from=cpython_builder /usr/local/cpython-${_docker_cpython_version} /usr/local/cpython-${_docker_cpython_version}
RUN mkdir -p /usr/local/cuda-12.3/nvvm/libdevice && ln -s /usr/local/cuda-12.3 /usr/local/cuda
COPY --from=ctk_builder /usr/local/cuda/nvvm/libdevice/libdevice.10.bc /usr/local/cuda-12.3/nvvm/libdevice

ENV PATH=/usr/local/cpython-${_docker_cpython_version}/bin:$PATH

RUN git clone https://private:oFqKOTthSTjcgzm9aksu@git.woa.com/wxg-natian/wnr.git
