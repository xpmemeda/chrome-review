cmake_minimum_required(VERSION 3.20)

project(example)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
include(ExternalProject)
### boost
set(BOOST_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/boost-1.86.0)
ExternalProject_Add(
  boost_project
  URL https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-b2-nodocs.tar.gz
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/boost-src
  BUILD_IN_SOURCE 1
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND <SOURCE_DIR>/bootstrap.sh --prefix=${BOOST_ROOT_DIR}
  BUILD_COMMAND
    <SOURCE_DIR>/b2
      install
      variant=release
      link=shared
      threading=multi
      runtime-link=shared
      cxxflags=-fPIC
      define=BOOST_STACKTRACE_USE_ADDR2LINE
      --with-stacktrace
      --with-random
      --with-system
      --with-thread
      --with-filesystem
      --with-chrono
      --with-atomic
      --with-date_time
      --with-regex
      -j32
  INSTALL_COMMAND ""
  STAMP_DIR ${BOOST_ROOT_DIR}
)
### grpc, includes protobuf.
set(GRPC_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/grpc-1.49.4)
ExternalProject_Add(
  grpc_project
  GIT_REPOSITORY https://github.com/grpc/grpc.git
  GIT_TAG v1.49.4
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/grpc-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/grpc-build
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND
    cmake
      -S <SOURCE_DIR>
      -B <BINARY_DIR>
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${GRPC_ROOT_DIR}
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install
  STAMP_DIR ${GRPC_ROOT_DIR}
)
### cpprestsdk
set(CPPRESTSDK_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/cpprestsdk-2.10.19)
ExternalProject_Add(
  cpprestsdk_project
  DEPENDS boost_project
  GIT_REPOSITORY https://github.com/microsoft/cpprestsdk.git
  GIT_TAG v2.10.19
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/cpprestsdk-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/cpprestsdk-build
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND
    cmake
      -S <SOURCE_DIR>
      -B <BINARY_DIR>
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${CPPRESTSDK_ROOT_DIR}
      -DCMAKE_PREFIX_PATH=${BOOST_ROOT_DIR}
      -DCPPREST_EXCLUDE_WEBSOCKETS=ON
      -DCMAKE_CXX_FLAGS="-Wno-error=format-truncation"
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install
  STAMP_DIR ${CPPRESTSDK_ROOT_DIR}
)
### etcd
set(ETCD_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/etcd-cpp-apiv3-0.15.4)
ExternalProject_Add(
  etcd_cpp_apiv3_project
  DEPENDS grpc_project cpprestsdk_project
  GIT_REPOSITORY https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git
  GIT_TAG v0.15.4
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/etcd-cpp-apiv3-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/etcd-cpp-apiv3-build
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND
    CMAKE_PREFIX_PATH=${GRPC_ROOT_DIR}:${CPPRESTSDK_ROOT_DIR}
    cmake
      -S <SOURCE_DIR>
      -B <BINARY_DIR>
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${ETCD_ROOT_DIR}
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install
  STAMP_DIR ${ETCD_ROOT_DIR}
)


add_executable(example example.cpp)
add_dependencies(example etcd_cpp_apiv3_project)
target_include_directories(example PUBLIC ${ETCD_ROOT_DIR}/include ${CPPRESTSDK_ROOT_DIR}/include)
target_link_directories(example PUBLIC ${ETCD_ROOT_DIR}/lib64 ${BOOST_ROOT_DIR}/lib)
target_link_libraries(example etcd-cpp-api)