add_library(rnpz rnpz.h rnpz.cpp)
target_link_libraries(rnpz z)
set_target_properties(rnpz PROPERTIES PUBLIC_HEADER rnpz.h)

add_executable(main main.cpp)
target_link_libraries(main rnpz)

install(
  TARGETS rnpz
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include
)