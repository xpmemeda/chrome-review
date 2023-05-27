cmake_minimum_required(VERSION 3.10)
project(hello-world)

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

add_subdirectory(code-generation)
add_dependencies(${PROJECT_NAME} generator)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/code-generation)