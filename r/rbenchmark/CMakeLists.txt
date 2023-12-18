add_library(rbenchmark_utils ${CMAKE_CURRENT_SOURCE_DIR}/rbenchmark_utils.cpp)
set_target_properties(rbenchmark_utils PROPERTIES PUBLIC_HEADER rbenchmark_utils.h)

install(
  TARGETS rbenchmark_utils
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include
)