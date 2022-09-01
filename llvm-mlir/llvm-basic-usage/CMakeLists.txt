cmake_minimum_required(VERSION 3.14)
project(main)

find_package(LLVM REQUIRED)
include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

add_executable(main main.cpp)
llvm_map_components_to_libnames(LLVM_LIBS support core nativecodegen x86asmparser orcjit)
target_link_libraries(main ${LLVM_LIBS})
