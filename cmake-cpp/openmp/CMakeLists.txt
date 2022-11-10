cmake_minimum_required(VERSION 3.10)

project(hello-world)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(${PROJECT_NAME} main.cpp)
