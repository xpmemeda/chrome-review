cmake_minimum_required(VERSION 3.10)

project(hello_world)
enable_language(C ASM)

SET(CMAKE_ASM_FLAGS "${CFLAGS} -x assembler-with-cpp")

execute_process(COMMAND ${CMAKE_C_COMPILER} -S ${CMAKE_CURRENT_SOURCE_DIR}/sum_array.cpp -o ${CMAKE_CURRENT_BINARY_DIR}/sum_array.s)

file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${CMAKE_CURRENT_BINARY_DIR}/sum_array.s)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
