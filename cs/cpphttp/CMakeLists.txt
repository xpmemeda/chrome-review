cmake_minimum_required(VERSION 3.14)

project(cpphttp)

include(ExternalProject)

ExternalProject_Add(
    cpp-httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG master
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external_projects/cpp-httplib
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)

ExternalProject_Add(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG master
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external_projects/nlohmann_json
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)


include_directories(
    ${CMAKE_CURRENT_BINARY_DIR}/external_projects/cpp-httplib
    ${CMAKE_CURRENT_BINARY_DIR}/external_projects/nlohmann_json/include
)

add_executable(${PROJECT_NAME} main.cpp)

add_dependencies(${PROJECT_NAME} cpp-httplib nlohmann_json)

target_link_libraries(${PROJECT_NAME} pthread)
