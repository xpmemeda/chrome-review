CMAKE_HOME=$HOME/local/cmake
GCC_HOME=$HOME/local/gcc
GDB_HOME=$HOME/local/gdb
CPYTHON_HOME=$HOME/local/cpython
CUDA_HOME=$HOME/local/cuda
LLVM_HOME=$HOME/local/llvm
ONEDNN_HOME=$HOME/local/oneDNN
PROTOBUF_HOME=$HOME/local/protobuf
VECTORCLASS_HOME=$HOME/local/vectorclass
CUDNN_HOME=$HOME/local/cudnn
PATCHELF_HOME=$HOME/local/patchelf
OPENMPI_HOME=$HOME/local/openmpi
GTEST_HOME=$HOME/local/googletest
RUSTUP_HOME=$HOME/local/rustup
CARGO_HOME=$HOME/local/cargo

FARM_HASH_HOME=$HOME/local/wnr-deps/farmhash
HIGHWAY_HASH_HOME=$HOME/local/wnr-deps/highwayhash
MLAS_HOME=$HOME/local/wnr-deps/mlas
SVML_HOME=$HOME/local/wnr-deps/svml

wnr_dep_libs=(
    "$CMAKE_HOME"
    "$GCC_HOME"
    "$GDB_HOME"
    "$CPYTHON_HOME"
    "$CUDA_HOME"
    "$LLVM_HOME"
    "$ONEDNN_HOME"
    "$PROTOBUF_HOME"
    "$VECTORCLASS_HOME"
    "$FARM_HASH_HOME"
    "$HIGHWAY_HASH_HOME"
    "$MLAS_HOME"
    "$SVML_HOME"
    "$CUDNN_HOME"
    "$PATCHELF_HOME"
    "$OPENMPI_HOME"
    "$RUSTUP_HOME"
    "$CARGO_HOME"
)
for path in "${wnr_dep_libs[@]}"; do
    if [ ! -e "$path" ]; then
        echo -e "\033[31mPath does not exist: $path\033[0m"
    fi
done

export PATH=
export PATH=/usr/local/bin:/usr/bin:/usr/sbin:$PATH
export PATH=$CPYTHON_HOME/bin:$PROTOBUF_HOME/bin/:$CUDA_HOME/bin:$CMAKE_HOME/bin:$GDB_HOME/bin:$GCC_HOME/bin:$PATH
export PATH=$PATCHELF_HOME/bin:$PATH
export PATH=$CARGO_HOME/bin:$OPENMPI_HOME/bin:$PATH
export PATH=$HOME/.ft:$HOME/local/patchbuild:$PATH
export CMAKE_PREFIX_PATH=
export CMAKE_PREFIX_PATH=$HOME/local:$CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH=$ONEDNN_HOME:$PROTOBUF_HOME:$LLVM_HOME:$CMAKE_PREFIX_PATH

export CUDNN_HOME FARM_HASH_HOME HIGHWAY_HASH_HOME MLAS_HOME SVML_HOME
export VECTORCLASS_HOME DNNL_HOME=$ONEDNN_HOME
export GTEST_HOME

export CC=$(which gcc)
export CXX=$(which g++)

gcc_version=$(gcc --version 2>&1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')
python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null)
export PS1=$'\033[0;32m'"(python-$python_version) (gcc-$gcc_version)"$'\033[0m'"${PS1-}"
export PYTHON_SITELIB=$CPYTHON_HOME/lib/python${python_version}/site-packages

export CPATH=$CUDNN_HOME/include:$OPENMPI_HOME/include
# It means that the current directory is included if the path ends with ":".
export LIBRARY_PATH=
export LIBRARY_PATH=/usr/local/lib64:/usr/lib64
export LIBRARY_PATH=$GCC_HOME/lib64:$LIBRARY_PATH
export LIBRARY_PATH=$OPENMPI_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=
export LD_LIBRARY_PATH=/usr/local/lib64:/usr/lib64
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$CUDA_HOME/lib64:$GCC_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENMPI_HOME/lib:$LD_LIBRARY_PATH
