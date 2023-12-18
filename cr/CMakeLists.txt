cmake_minimum_required(VERSION 3.20)

project(cr)

file(GLOB SOURCE_FILES "*.cpp")
list(FILTER SOURCE_FILES EXCLUDE REGEX "main.cpp")
add_library(${PROJECT_NAME} ${SOURCE_FILES})

add_executable(main "main.cpp")
target_link_libraries(main ${PROJECT_NAME})

file(GLOB HEADER_FILES "*.h")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include/chrome-review
)
