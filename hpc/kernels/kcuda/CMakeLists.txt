cmake_minimum_required(VERSION 3.20)

project(kernels LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build : None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()
message("* Current build type is : ${CMAKE_BUILD_TYPE}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)  # compile
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG v3.8.0
)
FetchContent_Populate(cutlass)  # not compile.
include_directories(${cutlass_SOURCE_DIR}/include)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(Utils)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDAToolkit REQUIRED)

# cudnn.
find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h PATHS ${CUDAToolkit_INCLUDE_DIR} $ENV{HOME}/local/cudnn PATH_SUFFIXES include REQUIRED)
find_library(CUDNN_LIBRARY NAMES cudnn PATHS ${CUDAToolkit_LIBRARY_DIR} $ENV{HOME}/local/cudnn PATH_SUFFIXES lib lib64 REQUIRED)
include_directories(${CUDNN_INCLUDE_DIR})

get_torch_cmake_prefix_path(TORCH_CMAKE_PREFIX_PATH 2.0.0)
find_package(Torch PATHS ${TORCH_CMAKE_PREFIX_PATH} REQUIRED)

add_executable(example example.cu)
target_link_libraries(example PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cublas)
target_link_libraries(example PRIVATE ${CUDNN_LIBRARY})

file(GLOB MODULE_SOURCE_FILES "*.cpp" "*.cu")
list(REMOVE_ITEM MODULE_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/example.cu")
pybind11_add_module(${PROJECT_NAME} ${MODULE_SOURCE_FILES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 80 90)

target_compile_options(${PROJECT_NAME} PRIVATE -O2)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fpch-preprocess,-Wall,-Wextra>)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cublas)
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDNN_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
# Resolve: undefined symbol: _ZN8pybind116detail11type_casterIN2at6TensorEvE4loadENS_6handleEb
target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")

add_custom_target(create_symlink ALL)
add_custom_target(remove_old_symlink COMMAND ${CMAKE_COMMAND} -E rm -f ${CMAKE_CURRENT_SOURCE_DIR}/*.so)
add_dependencies(create_symlink remove_old_symlink)
add_custom_target(
  create_symlink_${PROJECT_NAME}
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:${PROJECT_NAME}>
)
add_dependencies(create_symlink create_symlink_${PROJECT_NAME})
