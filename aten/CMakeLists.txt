cmake_minimum_required(VERSION 3.20)

project(aten LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build : None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()
message("* Current build type is : ${CMAKE_BUILD_TYPE}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
endif()

find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})
find_package(Torch PATHS ${Python_SITEARCH}/torch/share/cmake REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

include(FetchContent)

FetchContent_Declare(
  nvtx
  GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
  GIT_TAG v3.3.0
)
FetchContent_MakeAvailable(nvtx)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC ${TORCH_LIBRARIES} ${Python_SITEARCH}/torch/lib/libtorch_python.so)
target_link_libraries(${PROJECT_NAME} PUBLIC Python::Python)
target_link_libraries(${PROJECT_NAME} PUBLIC nvtx3-cpp)
