# project ----------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
project(example CXX)
set(CMAKE_CXX_STANDARD 17)

# deps -------------------------------------------------------------
### comm
add_subdirectory(comm)

# targets ----------------------------------------------------------
add_executable(example example.cpp zmqbackend.cpp)
target_link_libraries(example PRIVATE comm)
