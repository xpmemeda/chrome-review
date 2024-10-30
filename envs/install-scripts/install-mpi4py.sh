#!/bin/bash

OPENMPI_HOME=${HOME}/local/openmpi

if [ ! -d ${OPENMPI_HOME} ]; then
    echo "[ERR]openmpi not found"
    exit 1
fi

if ! echo $LIBRARY_PATH | grep -q "${OPENMPI_HOME}/lib"; then
    echo "[ERR]openmpi env err, should be add to LIBRARY_PATH"
    exit 1
fi
if ! echo $LD_LIBRARY_PATH | grep -q "${OPENMPI_HOME}/lib"; then
    echo "[ERR]openmpi env err, should be add to LD_LIBRARY_PATH"
    exit 1
fi
if ! echo $CPATH | grep -q "${OPENMPI_HOME}/include"; then
    echo "[ERR]openmpi env err, should be add to CPATH"
    exit 1
fi
if ! echo $PATH | grep -q "${OPENMPI_HOME}/bin"; then
    echo "[ERR]openmpi env err, should be add to CPATH"
    exit 1
fi

export CC=${HOME}/local/openmpi/bin/mpicc
mpi_version=$(mpirun --version 2>&1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')
major_version=$(echo $mpi_version | cut -d. -f1)
echo ${major_version}

if [[ "${major_version}" -ne "4" ]]; then
    echo "[ERR]mpi version not match, should use mpi-4"
    exit 1
fi

python -m pip install mpi4py --no-cache-dir
