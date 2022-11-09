cmake_minimum_required(VERSION 3.15)

project(hello-world)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(${PROJECT_NAME} main.cpp)
find_library(DNNL_LIB NAMES dnnl PATHS $ENV{DNNL_HOME}/lib64)
target_link_libraries(${PROJECT_NAME} ${DNNL_LIB} dl)
target_include_directories(${PROJECT_NAME} PRIVATE $ENV{DNNL_HOME}/include)
