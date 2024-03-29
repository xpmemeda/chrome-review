cmake_minimum_required(VERSION 3.15)

project(main)
set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

add_executable(${PROJECT_NAME} main.cpp)

find_package(CUDA REQUIRED)
if (NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE FILEPATH "nvcc compiler path")
else()
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc CACHE FILEPATH "nvcc compiler path")
endif()
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
set(DEP_LIBRARYS "cudart" )
foreach(LIBRARY_NAME ${DEP_LIBRARYS})
    find_library(LIBRARY_PATH ${LIBRARY_NAME} ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    if ("${LIBRARY_PATH}" MATCHES ".*NOTFOUND")
        message(SEND_ERROR "Library ${LIBRARY_NAME} not found.")
    endif()
    target_link_libraries(${PROJECT_NAME} ${LIBRARY_PATH})
    unset(LIBRARY_PATH CACHE)
    unset(LIBRARY_PATH)
endforeach()
