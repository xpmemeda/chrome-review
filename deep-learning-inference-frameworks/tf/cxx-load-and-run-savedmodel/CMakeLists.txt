cmake_minimum_required(VERSION 3.20)

project(cxx-load-and-run-savedmodel)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.5)
    message(FATAL_ERROR "Insufficient gcc version, 10.5 or later required")
  endif()
endif()
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} main.cpp)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

target_include_directories(${PROJECT_NAME} PRIVATE "$ENV{HOME}/local/r/include" "${Python_SITELIB}/tensorflow/include")
target_link_directories(${PROJECT_NAME} PRIVATE "$ENV{HOME}/local/r/lib")
target_link_libraries(${PROJECT_NAME} PRIVATE "${Python_SITELIB}/tensorflow/libtensorflow_cc.so.2" "${Python_SITELIB}/tensorflow/libtensorflow_framework.so.2")
target_link_libraries(${PROJECT_NAME} PRIVATE rnpz rbenchmark_utils)
target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
