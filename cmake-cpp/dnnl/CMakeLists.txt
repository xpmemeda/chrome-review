cmake_minimum_required(VERSION 3.15)

project(hello-world)

find_library(DNNL_LIB NAMES dnnl PATHS $ENV{DNNL_HOME}/lib64)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${DNNL_LIB})
target_include_directories(${PROJECT_NAME} PRIVATE $ENV{DNNL_HOME}/include)