cmake_minimum_required(VERSION 3.20)

project(cr)

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

include(utils.cmake)

protobuf_generate_cpp(PB_CPP_SRCS PB_CPP_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/cr.proto)
protobuf_generate_python(PB_PYTHON ${CMAKE_CURRENT_SOURCE_DIR}/cr.proto)

get_relative_path(relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${PB_PYTHON})

add_custom_target(
    create_symlink
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${relative_path} ${CMAKE_CURRENT_SOURCE_DIR}/cr_pb2.py
    DEPENDS ${PB_PYTHON}
)
add_custom_target(dummy ALL DEPENDS create_symlink)

add_executable(${PROJECT_NAME} main.cpp ${PB_CPP_SRCS} ${PB_CPP_HDRS})
target_link_libraries(${PROJECT_NAME} ${Protobuf_LIBRARIES})

add_custom_target(
    clear
    COMMAND rm -rf "${CMAKE_CURRENT_SOURCE_DIR}/cr_pb2.py"
    COMMAND rm -rf "${CMAKE_CURRENT_SOURCE_DIR}/*.pb"
)