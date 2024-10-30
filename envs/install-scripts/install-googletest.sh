#!/bin/bash

if [ -z "${TMPDIR}" ]; then
  echo "TMPDIR is not set."
  exit 1
fi

VERSION="1.12.1"
INSTALL_PATH=${HOME}/local/googletest-${VERSION}

if [ -d ${INSTALL_PATH} ]; then
  echo "${INSTALL_PATH} has exist"
  echo 0
fi

TMPPATH=${TMPDIR}/googletest-${VERSION}.zip
if [ ! -f ${TMPPATH} ]; then
  wget https://github.com/google/googletest/archive/refs/tags/release-${VERSION}.zip -O ${TMPPATH}
  if [ $? -ne 0 ]; then
    echo "download to ${TMPPATH} err"
    exit 1
  fi
fi

ZIPROOT=$(unzip -l "$TMPPATH" | awk 'NR>3 {print $4}' | grep -o '^[^/]*/' | sort -u | head -n 1)
cd ${TMPDIR} && unzip ${TMPPATH} -d googletest-${VERSION} && cd googletest-${VERSION}/${ZIPROOT}

cmake . -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} -DCMAKE_BUILD_TYPE=Release -DBUILD_GMOCK=OFF -B build
cmake --build build -j && cmake --install build

if [ $? -eq 0 ]; then
  echo "Installation completed!"
  rm -rf ${TMPDIR}/googletest-${VERSION}.zip ${TMPDIR}/googletest-${VERSION}
fi