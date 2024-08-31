cmake_minimum_required(VERSION 3.20)

project(load_and_run_onnx_model)

add_executable(${PROJECT_NAME} main.cpp)

if (NOT TENSORRT_HOME)
    if (NOT DEFINED ENV{TENSORRT_HOME})
        message(FATAL_ERROR "Cannot find TensorRT, please use -D or enveriment variables to define TENSORRT_HOME")
    endif()
    set(TENSORRT_HOME "$ENV{TENSORRT_HOME}" CACHE PATH "TensorRT home" FORCE)
endif()
target_include_directories(${PROJECT_NAME} PRIVATE ${TENSORRT_HOME}/include)
target_link_directories(${PROJECT_NAME} PRIVATE ${TENSORRT_HOME}/lib)
target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvonnxparser)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas CUDA::cudart CUDA::cuda_driver)

if (NOT CR_HOME)
    if (NOT DEFINED ENV{CR_HOME})
        message(FATAL_ERROR "Cannot find CR lib, please use -D or enveriment variables to define CR_HOME")
    endif()
    set(CR_HOME "$ENV{CR_HOME}" CACHE PATH "CR home" FORCH)
endif()
target_include_directories(${PROJECT_NAME} PRIVATE "${CR_HOME}/include")
target_link_directories(${PROJECT_NAME} PRIVATE "${CR_HOME}/lib")
target_link_libraries(${PROJECT_NAME} PRIVATE cr z pthread)
