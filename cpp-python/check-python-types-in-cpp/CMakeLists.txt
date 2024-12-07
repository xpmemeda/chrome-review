cmake_minimum_required(VERSION 3.10)

project(hello_world)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.10.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(${PROJECT_NAME} main.cpp)

add_custom_command(
  TARGET ${PROJECT_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.cpython-36m-x86_64-linux-gnu.so
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.cpython-36m-x86_64-linux-gnu.so
)