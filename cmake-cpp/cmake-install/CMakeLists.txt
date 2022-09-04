cmake_minimum_required(VERSION 3.10)

project(hello-world-cmake-install VERSION 1.0 DESCRIPTION "hello-world-cmake-install-example")

file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE HDR_FILES "include/*.h")

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
# # $<BUILD_INTERFACE:...>: For self build
# # $<INSTALL_INTERFACE:...>: For library users, automaticly include_directories(...)
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER ${HDR_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

# install hdrs, libs and executables to specified dir
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include
)

# generate ``{PROJECT_NAME}-config.cmake``, which
#   1) record the targets to be installed
#   2) in other project: find_package will include this file
install(EXPORT ${PROJECT_NAME} FILE ${PROJECT_NAME}-config.cmake DESTINATION cmake)
