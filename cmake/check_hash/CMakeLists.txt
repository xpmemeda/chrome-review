cmake_minimum_required(VERSION 3.5)
project(CHECK_HASH)

function(check_hash SRC_FILES)
    execute_process(COMMAND ${CMAKE_COMMAND} -E md5sum ${SRC_FILES} OUTPUT_VARIABLE HASH)
    message(STATUS ${HASH})
endfunction(check_hash SRC_FILES)

set(SRC_FILES ${PROJECT_SOURCE_DIR}/main.cpp ${PROJECT_SOURCE_DIR}/CMakeLists.txt)
message(${SRC_FILES})

check_hash("${SRC_FILES}")
