#!/bin/bash

set -e

echo "[INFO]current directory: $(pwd)"

ChangeDirectory() {
    echo "[INFO]cd $1"
    cd "$1"
    echo "[INFO]current directory: $(pwd)"
}

DownloadFile() {
    url="$1"
    output_path="$2"

    if [ -f "$output_path" ]; then
        echo "[INFO]reuse downloaded file: $output_path"
        return
    fi

    echo "[INFO]download $url to $output_path"
    if ! wget -O "$output_path" "$url"; then
        echo "[ERR]wget $url"
        rm -f "$output_path"
        exit 1
    fi
}

CPYTHON_VERSION=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --version|-v) CPYTHON_VERSION="$2"; shift ;;
        *) echo "unknown params: $1"; exit 1 ;;
    esac
    shift
done
if [ -z "$CPYTHON_VERSION" ]; then
    echo "usage: $0 --version[-v] <version>"
    echo "example: $0 -v 3.10.18"
    exit 1
fi

INSTALL_DIR="$HOME/local/cpython-$CPYTHON_VERSION"
TEMP_DIR="$HOME/local/build.cpython-$CPYTHON_VERSION"
DOWNLOAD_DIR="${TMPDIR:-/tmp}"
OPENSSL_VERSION="1.1.1w"
OPENSSL_INSTALL_DIR="$HOME/local/openssl-$OPENSSL_VERSION"
OPENSSL_ARCHIVE="$DOWNLOAD_DIR/openssl-$OPENSSL_VERSION.tar.gz"
CPYTHON_ARCHIVE="$DOWNLOAD_DIR/v$CPYTHON_VERSION.zip"

InstallBuildDependencies() {
    if [ ! -r /etc/os-release ]; then
        echo "[ERR]unsupported system: /etc/os-release not found"
        exit 1
    fi

    . /etc/os-release
    os_id="${ID:-}"
    os_like="${ID_LIKE:-}"

    if [[ "$os_id" == "ubuntu" || "$os_like" == *"debian"* ]]; then
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            ca-certificates \
            libbz2-dev \
            libffi-dev \
            libgdbm-dev \
            libgmp-dev \
            liblzma-dev \
            libmpc-dev \
            libmpfr-dev \
            libncursesw5-dev \
            libreadline-dev \
            libsqlite3-dev \
            libssl-dev \
            perl \
            pkg-config \
            tk-dev \
            texinfo \
            unzip \
            wget \
            uuid-dev \
            xz-utils \
            zlib1g-dev
    elif [[ "$os_id" == "centos" || "$os_id" == "rhel" || "$os_id" == "rocky" || "$os_id" == "almalinux" ||
        "$os_like" == *"rhel"* || "$os_like" == *"fedora"* ]]; then
        if command -v dnf >/dev/null 2>&1; then
            sudo dnf groupinstall -y "Development Tools"
            sudo dnf install -y dnf-plugins-core || true
            sudo dnf install -y \
                bzip2-devel \
                ca-certificates \
                gdbm-devel \
                gmp-devel \
                libffi-devel \
                libmpc-devel \
                libuuid-devel \
                mpfr-devel \
                ncurses-devel \
                openssl-devel \
                readline-devel \
                sqlite-devel \
                tar \
                texinfo \
                tk-devel \
                unzip \
                wget \
                xz-devel \
                zlib-devel
            sudo dnf builddep -y python3 || true
        else
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y yum-utils || true
            sudo yum install -y \
                bzip2-devel \
                ca-certificates \
                gdbm-devel \
                gmp-devel \
                libffi-devel \
                libmpc-devel \
                libuuid-devel \
                mpfr-devel \
                ncurses-devel \
                openssl-devel \
                readline-devel \
                sqlite-devel \
                tar \
                texinfo \
                tk-devel \
                unzip \
                wget \
                xz-devel \
                zlib-devel
            sudo yum-builddep python3 -y || true
        fi
    else
        echo "[ERR]unsupported system: ID=$os_id ID_LIKE=$os_like"
        exit 1
    fi
}

InstallBuildDependencies

mkdir -p "$TEMP_DIR"
mkdir -p "$DOWNLOAD_DIR"
ChangeDirectory "$TEMP_DIR"

if [ ! -d "$OPENSSL_INSTALL_DIR" ]; then
    DownloadFile \
        "https://www.openssl.org/source/openssl-$OPENSSL_VERSION.tar.gz" \
        "$OPENSSL_ARCHIVE"
    tar -xzf "$OPENSSL_ARCHIVE"
    ChangeDirectory "openssl-$OPENSSL_VERSION"
    ./config --prefix="$OPENSSL_INSTALL_DIR"
    make -j && make install
fi

ChangeDirectory "$TEMP_DIR"

if [ ! -d "${INSTALL_DIR}" ]; then
    DownloadFile \
        "https://github.com/python/cpython/archive/refs/tags/v$CPYTHON_VERSION.zip" \
        "$CPYTHON_ARCHIVE"
    unzip "$CPYTHON_ARCHIVE"
    ChangeDirectory "cpython-$CPYTHON_VERSION"
    # NOTE: Use --enable-shared to generate libpython{MAJOR}.{MINOR}.so. see detail: ./configure --help
    # Build both shared and static libraries.
    # NOTE: Use --with-openssl and --with-openssl-rpath=auto to avoid:
    # warning: pip is configured with locations that require tls/ssl, however the ssl module in python is not available.
    mkdir build-shared
    ChangeDirectory "build-shared"
    ../configure \
        --prefix=$INSTALL_DIR \
        --enable-shared \
        --enable-optimizations \
        --with-openssl=$OPENSSL_INSTALL_DIR \
        --with-openssl-rpath=auto && make -j && make install
    ChangeDirectory ".."
    mkdir build-static
    ChangeDirectory "build-static"
    ../configure \
        --prefix=$INSTALL_DIR \
        --enable-optimizations \
        --with-openssl=$OPENSSL_INSTALL_DIR \
        --with-openssl-rpath=auto && make -j && make install
    ChangeDirectory "${INSTALL_DIR}"
    ln -s lib lib64
    ChangeDirectory "${INSTALL_DIR}/bin"
    ln -s python3 python
fi

ChangeDirectory "/"
rm -rf $TEMP_DIR
