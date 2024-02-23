cmake_minimum_required(VERSION 3.20)

project(main)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -O3")

add_executable(${PROJECT_NAME} main.cpp)

find_library(DNNL_LIB NAMES dnnl PATHS $ENV{DNNL_HOME}/lib64)
target_link_libraries(${PROJECT_NAME} PRIVATE ${DNNL_LIB} dl)
target_include_directories(${PROJECT_NAME} PRIVATE $ENV{DNNL_HOME}/include)

find_library(LIB_MLAS NAMES onnxruntime_mlas REQUIRED)
find_library(LIB_MLAS_COMMON NAMES onnxruntime_common REQUIRED)
find_library(LIB_MLAS_NSYNC NAMES nsync_cpp REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIB_MLAS} ${LIB_MLAS_COMMON} ${LIB_MLAS_NSYNC})

find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
find_package(cpuinfo REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE cpuinfo::cpuinfo)