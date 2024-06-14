cmake_minimum_required(VERSION 3.15)

project(shared)
set(CMAKE_CXX_STANDARD 17)

add_library(${PROJECT_NAME} SHARED shared.cpp)