cmake_minimum_required(VERSION 3.10)

project(hello-world)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} pthread)