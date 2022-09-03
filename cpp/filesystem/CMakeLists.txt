cmake_minimum_required(VERSION 3.10)

project(hello-world)

file(GLOB SOURCE_FILES *.cpp *.h)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} -lstdc++fs)
