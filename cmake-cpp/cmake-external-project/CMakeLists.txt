cmake_minimum_required(VERSION 3.10)

project(hello-world)

include(FetchContent)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 9.0.0
)
FetchContent_MakeAvailable(fmt)

# function(print_all_properties)
#   get_cmake_property(_variableNames VARIABLES)
#   list (SORT _variableNames)
#   foreach (_variableName ${_variableNames})
#       message(STATUS "${_variableName}=${${_variableName}}")
#   endforeach()
# endfunction(print_all_properties)

# function(print_target_properties target)
#     if(NOT TARGET ${target})
#       message(STATUS "There is no target named '${target}'")
#       return()
#     endif()

#     foreach(property ${CMAKE_PROPERTY_LIST})
#         string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

#         # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
#         if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
#             continue()
#         endif()

#         get_property(was_set TARGET ${target} PROPERTY ${property} SET)
#         if(was_set)
#             get_target_property(value ${target} ${property})
#             message("${target} ${property} = ${value}")
#         endif()
#     endforeach()
# endfunction()

file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} fmt)
