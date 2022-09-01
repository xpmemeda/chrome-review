cmake_minimum_required(VERSION 3.5)
project(example)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.10.0
)
FetchContent_MakeAvailable(pybind11)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
pybind11_add_module(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_custom_command(
    TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.cpython-36m-x86_64-linux-gnu.so
      ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.cpython-36m-x86_64-linux-gnu.so
)
