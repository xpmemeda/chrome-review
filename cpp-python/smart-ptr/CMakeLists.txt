cmake_minimum_required(VERSION 3.20)

project(smartptr)

find_package(Python 3 COMPONENTS Interpreter Development NumPy REQUIRED)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.10.0
)
FetchContent_MakeAvailable(pybind11)

file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
pybind11_add_module(${PROJECT_NAME} ${SOURCE_FILES})

add_custom_target(create_symlink ALL)
add_custom_target(
    create_symlink_${PROJECT_NAME}
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/*.so
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}/$<TARGET_FILE_NAME:${PROJECT_NAME}>
)
add_dependencies(create_symlink create_symlink_${PROJECT_NAME})
