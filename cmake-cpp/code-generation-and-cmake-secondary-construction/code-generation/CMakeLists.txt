set(GENERATOR_MAIN "generator_main")

add_executable(${GENERATOR_MAIN} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

set(out_head "add.h")
set(out_body "add.cpp")
add_custom_command(OUTPUT ${out_head} ${out_body} COMMAND ${GENERATOR_MAIN} ${out_head} ${out_body})
add_custom_target(generator DEPENDS ${out_head} ${out_body})