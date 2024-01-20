cmake_minimum_required(VERSION 3.10)

project(dlopen-demo)
set(CMAKE_CXX_STANDARD 17)

add_executable(main main.cpp)
target_link_libraries(main dl)

add_library(example SHARED examplelib.cpp)