cmake_minimum_required(VERSION 3.10)
project(main)

set (CMAKE_CXX_STANDARD 11)

include_directories(/usr/local/gtest/include/)
link_directories(/usr/local/gtest/lib64/)

add_executable(main main.cpp)
target_link_libraries(main pthread)
target_link_libraries(main gtest)
