cmake_minimum_required(VERSION 3.20)

project(cr)

file(GLOB SOURCE_FILES "*.cpp")
add_library(${PROJECT_NAME} ${SOURCE_FILES})

file(GLOB HEADER_FILES "*.h")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include/chrome-review
)
