cmake_minimum_required(VERSION 3.10)
project(main)

set (CMAKE_CXX_STANDARD 11)

include(ExternalProject)
EXTERNALPROJECT_ADD(
  libfmt
  URL https://github.com/fmtlib/fmt/archive/refs/tags/8.0.1.tar.gz
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libfmt
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/export
)

include_directories(${CMAKE_BINARY_DIR}/export/include)
link_directories(${CMAKE_BINARY_DIR}/export/lib64)

add_executable(main main.cpp)
target_link_libraries(main fmt)
add_dependencies(main libfmt)
