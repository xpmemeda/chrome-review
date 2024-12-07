cmake_minimum_required(VERSION 3.10)
project(example)

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

include_directories(/usr/include/python3.6m)

add_library(${PROJECT_NAME} SHARED main.cpp)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${PROJECT_NAME}"
        LINKER_LANGUAGE C
    )

add_custom_command(
    TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.so
      ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.so
)