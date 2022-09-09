cmake_minimum_required(VERSION 3.10)

project(hello-world-cmake-install VERSION 1.0 DESCRIPTION "hello-world-cmake-install-example")

# Get all propreties that cmake supports
if(NOT CMAKE_PROPERTY_LIST)
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)
    
    # Convert command output into a CMake list
    string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
endif()
    
function(print_properties)
    message("CMAKE_PROPERTY_LIST = ${CMAKE_PROPERTY_LIST}")
endfunction()
    
function(print_target_properties target)
    if(NOT TARGET ${target})
      message(STATUS "There is no target named '${target}'")
      return()
    endif()

    foreach(property ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
            continue()
        endif()

        get_property(was_set TARGET ${target} PROPERTY ${property} SET)
        if(was_set)
            get_target_property(value ${target} ${property})
            message("${target} ${property} = ${value}")
        endif()
    endforeach()
endfunction()



file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE HDR_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
message(STATUS "${HDR_FILES}")

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
# # $<BUILD_INTERFACE:...>: For self build
# # $<INSTALL_INTERFACE:...>: For library users, automaticly include_directories(...)
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HDR_FILES}")
# set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

print_target_properties(${PROJECT_NAME})

# install hdrs, libs and executables to specified dir
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  # PUBLIC_HEADER DESTINATION include
)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
  DESTINATION "include"
  FILES_MATCHING
    PATTERN "*.h"
)

# generate ``{PROJECT_NAME}-config.cmake``, which
#   1) record the targets to be installed
#   2) in other project: find_package will include this file
install(EXPORT ${PROJECT_NAME} FILE ${PROJECT_NAME}-config.cmake DESTINATION cmake)
