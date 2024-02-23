cmake_minimum_required(VERSION 3.20)

project(ort-load-and-run-model)

add_executable(${PROJECT_NAME} ort-load-and-run-model.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE "$ENV{HOME}/local/r/include" "$ENV{HOME}/local/onnxruntime/include")
target_link_directories(${PROJECT_NAME} PRIVATE "$ENV{HOME}/local/r/lib" "$ENV{HOME}/local/onnxruntime/lib")
target_link_libraries(${PROJECT_NAME} onnxruntime rnpz z pthread)
