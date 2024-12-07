cmake_minimum_required(VERSION 3.10)

project(hello_world)

find_package(Python3 COMPONENTS Interpreter NumPy REQUIRED)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.10.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE Python3::NumPy)

add_custom_command(
  TARGET ${PROJECT_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_SOURCE_DIR}/$<TARGET_FILE_NAME:${PROJECT_NAME}>
)