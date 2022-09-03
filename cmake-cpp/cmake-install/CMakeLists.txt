cmake_minimum_required(VERSION 3.10)

project(add VERSION 1.0)

file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE HDR_FILES "include/*.h")

add_library(add SHARED ${SRC_FILES})
# $<BUILD_INTERFACE:...>: For self build
# $<INSTALL_INTERFACE:...>: For library users, automaticly include_directories(...)
target_include_directories(add PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
set_target_properties(add PROPERTIES PUBLIC_HEADER ${HDR_FILES})

# install hdrs and lib
install(TARGETS add EXPORT ${PROJECT_NAME}
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include)

# install cmake file
install(EXPORT ${PROJECT_NAME} DESTINATION cmake)


# echo `cat add-config.cmake.in` > add-config.cmake
configure_file(add-config.cmake.in ${PROJECT_NAME}-config.cmake @ONLY)
# generate version info
include(CMakePackageConfigHelpers)
write_basic_package_version_file(${PROJECT_NAME}-config-version.cmake VERSION ${PACKAGE_VERSION} COMPATIBILITY AnyNewerVersion)
# install cmake config file
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake" 
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
        DESTINATION cmake)
