cmake_minimum_required(VERSION 3.20)

project(rediscpp)

set(CMAKE_CXX_STANDARD 17)

include(ExternalProject)
### hiredis
set(HIREDIS_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/hiredis-1.2.0)
ExternalProject_Add(
  hiredis_project
  GIT_REPOSITORY https://github.com/redis/hiredis.git
  GIT_TAG v1.2.0
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/hiredis-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/hiredis-build
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND
    cmake
      -S <SOURCE_DIR>
      -B <BINARY_DIR>
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DCMAKE_INSTALL_PREFIX=${HIREDIS_ROOT_DIR}
      -DBUILD_SHARED_LIBS=OFF
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install
)

add_library(rediscpp rediscpp.cpp)
add_dependencies(rediscpp hiredis_project)
target_include_directories(rediscpp PUBLIC ${HIREDIS_ROOT_DIR}/include/hiredis)
target_link_directories(rediscpp PUBLIC ${HIREDIS_ROOT_DIR}/lib64)
target_link_libraries(rediscpp PUBLIC hiredis)

add_executable(example example.cpp)
target_link_libraries(example rediscpp)
