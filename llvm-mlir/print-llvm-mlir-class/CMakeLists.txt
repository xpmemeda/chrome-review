cmake_minimum_required(VERSION 3.10)
project(hello-world LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 14 CACHE STRINGS "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
# set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
# set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

message(STATUS "LLVM_CMAKE_DIR: ${LLVM_CMAKE_DIR}")
message(STATUS "MLIR_CMAKE_DIR: ${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM_LIBRARY_DIRS: ${LLVM_LIBRARY_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
message(STATUS "MLIR_LIBRARY_DIRS: ${MLIR_LIBRARY_DIRS}")
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

message(STATUS "LLVM_BUILD_LIBRARY_DIR: ${LLVM_BUILD_LIBRARY_DIR}")
link_directories(${LLVM_BUILD_LIBRARY_DIR})

message(STATUS "LLVM_DEFINITIONS: ${LLVM_DEFINITIONS}")
add_definitions(${LLVM_DEFINITIONS})


# compared to add_executable, add_llvm_executable have some default options.
include_directories($ENV{GTEST_HOME}/include/)
link_directories($ENV{GTEST_HOME}/lib64/)

file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_llvm_executable(hello-world ${SOURCE_FILES})

add_dependencies(${PROJECT_NAME} demo_dialect)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
message(STATUS "dialect_libs: ${dialect_libs}")
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
message(STATUS "conversion_libs: ${conversion_libs}")
set(LIBS ${dialect_libs} ${conversion_libs})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRef
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTransforms
    )

target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
target_link_libraries(${PROJECT_NAME} PRIVATE gtest)
