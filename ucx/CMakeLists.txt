cmake_minimum_required(VERSION 3.18)

# Project settings.
project(ucx)
set(CMAKE_CXX_STANDARD 17)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Requirements.
find_package(CUDAToolkit REQUIRED)
link_libraries(CUDA::cudart)

include(ExternalProject)
set(UCX_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/ucx-1.19.0)
ExternalProject_Add(
    ucx_project
    GIT_REPOSITORY https://github.com/openucx/ucx.git
    GIT_TAG v1.19.0
    SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/ucx-src
    BUILD_IN_SOURCE 1
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND autogen.sh && ./contrib/configure-release-mt --enable-shared --disable-static --disable-doxygen-doc --enable-optimizations --enable-cma --enable-devel-headers --with-cuda=/usr/local/cuda --with-verbs --with-dm --prefix=${UCX_ROOT_DIR}
    BUILD_COMMAND make -j
    INSTALL_COMMAND make install
    STAMP_DIR ${UCX_ROOT_DIR}
)
include_directories(${UCX_ROOT_DIR}/include)
link_directories(${UCX_ROOT_DIR}/lib)

add_subdirectory(comm)

add_executable(svr svr.cpp ucxtx.cpp)
add_dependencies(svr ucx_project)
target_link_libraries(svr comm)
target_link_libraries(svr ucp ucs uct)

add_executable(cli cli.cpp ucxtx.cpp)
add_dependencies(cli ucx_project)
target_link_libraries(cli comm)
target_link_libraries(cli ucp ucs uct)

add_executable(example example.cpp ucxtx.cpp ucxbackend.cpp)
add_dependencies(example ucx_project)
target_link_libraries(example comm)
target_link_libraries(example ucp ucs uct)
