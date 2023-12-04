cmake_minimum_required(VERSION 3.20)

project(kernels LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)


find_package(Python COMPONENTS Interpreter Development REQUIRED)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(Utils)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB SOURCE_FILES "*.cpp" "kernels/*.cpp" "kernels/*.cu")
pybind11_add_module(${PROJECT_NAME} ${SOURCE_FILES})
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>) # ncu.

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cublas)
get_torch_cmake_prefix_path(TORCH_CMAKE_PREFIX_PATH 2.0.0)
find_package(Torch PATHS ${TORCH_CMAKE_PREFIX_PATH} REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")

add_custom_target(create_symlink ALL)
add_custom_target(remove_old_symlink COMMAND ${CMAKE_COMMAND} -E rm -f ${CMAKE_CURRENT_SOURCE_DIR}/*.so)
add_dependencies(create_symlink remove_old_symlink)
add_custom_target(
  create_symlink_${PROJECT_NAME}
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:${PROJECT_NAME}>
)
add_dependencies(create_symlink create_symlink_${PROJECT_NAME})
