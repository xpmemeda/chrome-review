cmake_minimum_required(VERSION 3.20)
project(ccl LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(${PROJECT_NAME} nccl.cu)


find_package(CUDAToolkit REQUIRED)

include(ExternalProject)
ExternalProject_Add(
    boost_project
    URL https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/boost
    CONFIGURE_COMMAND ./bootstrap.sh
    BUILD_COMMAND ""
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ./b2 -j 32 --with-program_options --with-system link=static cxxflags=-fPIC variant=release threading=multi runtime-link=static install --prefix=${CMAKE_BINARY_DIR}/third_party/boost
)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR}/third_party/boost/include)
target_link_directories(${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR}/third_party/boost/lib64)
target_link_directories(${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR}/third_party/boost/lib)

set(NCCL_ROOT "/usr" CACHE PATH "Path to NCCL installation")
find_library(NCCL_LIBRARY 
  NAMES nccl
  PATHS ${NCCL_ROOT}/lib /usr/lib /usr/local/cuda/targets/x86_64-linux/lib
  REQUIRED
)
find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  PATHS ${NCCL_ROOT}/include /usr/include /usr/local/cuda/targets/x86_64-linux/include
  REQUIRED
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${NCCL_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  CUDA::cudart         # CUDA Runtime
  ${NCCL_LIBRARY}      # NCCL
)


add_dependencies(${PROJECT_NAME} boost_project)
target_link_libraries(${PROJECT_NAME} PRIVATE boost_program_options)
