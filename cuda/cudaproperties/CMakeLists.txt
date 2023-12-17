cmake_minimum_required(VERSION 3.15)

project(main)
set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

add_executable(${PROJECT_NAME} main.cpp)

find_package(CUDAToolkit)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cublas)
target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
