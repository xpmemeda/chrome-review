cmake_minimum_required(VERSION 3.14)
project(hello-world)

find_package(LLVM REQUIRED)
include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
llvm_map_components_to_libnames(LLVM_LIBS support core nativecodegen x86asmparser orcjit)
target_link_libraries(${PROJECT_NAME} ${LLVM_LIBS})
