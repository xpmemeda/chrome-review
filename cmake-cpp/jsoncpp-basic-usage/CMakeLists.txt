cmake_minimum_required(VERSION 3.10)

project(main)

include_directories(jsoncpp)

add_executable(main main.cpp jsoncpp/jsoncpp.cpp)
