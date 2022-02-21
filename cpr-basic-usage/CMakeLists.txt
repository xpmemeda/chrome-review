cmake_minimum_required(VERSION 3.10)

project(main)

include(FetchContent)

FetchContent_Declare(cpr GIT_REPOSITORY https://github.com/libcpr/cpr.git
                         GIT_TAG beb9e98806bb84bcc130a2cebfbcbbc6ce62b335)
FetchContent_MakeAvailable(cpr)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE cpr::cpr)
