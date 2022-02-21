cmake_minimum_required(VERSION 3.10)
project(main)

set (CMAKE_CXX_STANDARD 11)

# The ABI for ``std::string`` has changed in GCC5, which caused an undefined reference error.
# This flag will force GCC to use the old ABI version.
# Another resolution: rebuild the protoc.
# Detail: https://stackoverflow.com/questions/30124264/undefined-reference-to-googleprotobufinternalempty-string-abicxx11
# set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
# ${PROTO_SRCS}: *.pb.cpp
# ${PROTO_HDRS}: *.pb.h
# *.pb.h and *.pb.cpp will be generated in ${CMAKE_CURRENT_BINARY_DIR}
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS addressbook.proto)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(main main.cpp ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(main ${Protobuf_LIBRARIES})
