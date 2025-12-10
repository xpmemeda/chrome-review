cmake_minimum_required(VERSION 3.20)

project(experiment CXX CUDA)

add_executable(experiment_cxx experiment.cpp)
add_executable(experiment_cu experiment.cu)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(experiment_cu CUDA::cudart)
