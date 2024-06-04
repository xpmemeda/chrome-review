cmake_minimum_required(VERSION 3.20)

project(main)

set(CMAKE_CXX_STANDARD 17)

file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} stdc++fs)
