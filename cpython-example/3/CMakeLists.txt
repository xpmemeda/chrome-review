cmake_minimum_required(VERSION 3.10)
project(example)

add_subdirectory(pybind11)

pybind11_add_module(${PROJECT_NAME} main.cpp)

add_custom_command(
    TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.cpython-36m-x86_64-linux-gnu.so
      ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.cpython-36m-x86_64-linux-gnu.so
)
