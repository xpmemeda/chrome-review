# project.
# ----------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(example CXX)
set(CMAKE_CXX_STANDARD 17)

# deps
# ----------------------------------------------------------
include(ExternalProject)
# python3
find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})
# torch
find_package(Torch PATHS ${Python_SITEARCH}/torch/share/cmake REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
# ucx-1.20.x
set(UCX_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/ucx-1.20.x)
ExternalProject_Add(
    ucx_project
    GIT_REPOSITORY https://github.com/openucx/ucx.git
    GIT_TAG v1.20.x
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
# nixl-0.8.0
set(NIXL_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/nixl-0.8.0)
ExternalProject_Add(
    nixl_project
    GIT_REPOSITORY https://github.com/ai-dynamo/nixl.git
    GIT_TAG 0.8.0
    SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/nixl-src
    BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/nixl-build
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND meson setup <BINARY_DIR> <SOURCE_DIR> -Dcpp_link_args=-Wl,-rpath,$ORIGIN:${UCX_ROOT_DIR}/lib -Ducx_path=${UCX_ROOT_DIR} --prefix ${NIXL_ROOT_DIR}
    BUILD_COMMAND ninja
    INSTALL_COMMAND ninja install
    STAMP_DIR ${NIXL_ROOT_DIR}
)
add_dependencies(nixl_project ucx_project)
include_directories(${NIXL_ROOT_DIR}/include)
link_directories(${NIXL_ROOT_DIR}/lib64)
# comm
add_subdirectory(comm)

# targets.
# ----------------------------------------------------------
add_executable(example example.cpp nixlbackend.cpp)
add_dependencies(example ucx_project nixl_project)
target_link_libraries(example PRIVATE nixl nixl_build serdes)
target_link_libraries(example PRIVATE comm)
target_link_libraries(example PRIVATE ${TORCH_LIBRARIES} ${Python_SITEARCH}/torch/lib/libtorch_python.so Python::Python)
