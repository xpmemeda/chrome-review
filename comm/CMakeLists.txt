cmake_minimum_required(VERSION 3.20)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  project(comm LANGUAGES CXX)
endif()

option(COMM_BUILD_SHARED "Build comm as a shared library" OFF)

include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2)
FetchContent_Declare(
  libzmq
  GIT_REPOSITORY https://github.com/zeromq/libzmq.git
  GIT_TAG v4.3.5)
FetchContent_Declare(
  cppzmq
  GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
  GIT_TAG v4.9.0)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.14.1)
FetchContent_MakeAvailable(json libzmq cppzmq spdlog)

file(GLOB_RECURSE COMM_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx")

if(COMM_BUILD_SHARED)
  add_library(comm SHARED ${COMM_SOURCES})
else()
  add_library(comm STATIC ${COMM_SOURCES})
endif()

target_link_libraries(comm PUBLIC nlohmann_json::nlohmann_json libzmq cppzmq
                                  spdlog::spdlog_header_only)

target_include_directories(comm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set_target_properties(comm PROPERTIES ARCHIVE_OUTPUT_DIRECTORY
                                        "${CMAKE_BINARY_DIR}/lib")
endif()
