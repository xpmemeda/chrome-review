add_library(rstl rstl.h rstl.cpp)
set_target_properties(rstl PROPERTIES PUBLIC_HEADER rstl.h)

install(
  TARGETS rstl
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include
)