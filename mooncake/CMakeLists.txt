cmake_minimum_required(VERSION 3.20)
project(mooncake)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
include(ExternalProject)
### cpp-httplib
set(CPPHTTPLIB_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/cpp-httplib)
ExternalProject_Add(
  cpp_httplib_project
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.28.0
  SOURCE_DIR ${CPPHTTPLIB_ROOT_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)
### glog
set(GLOG_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/glog)
ExternalProject_Add(
  glog_project
  GIT_REPOSITORY https://github.com/google/glog.git
  GIT_TAG v0.4.0
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/glog-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/glog-build
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND cmake -S <SOURCE_DIR> -B <BINARY_DIR> -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${GLOG_ROOT_DIR} -DBUILD_SHARED_LIBS=ON
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install -j
  STAMP_DIR ${GLOG_ROOT_DIR}/cmake-stamps
)
### gflags
set(GFLAGS_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/gflags)
ExternalProject_Add(
  gflags_project
  GIT_REPOSITORY https://github.com/gflags/gflags.git
  GIT_TAG v2.2.2
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/gflags-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/gflags-build
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND cmake -S <SOURCE_DIR> -B <BINARY_DIR> -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${GFLAGS_ROOT_DIR} -DBUILD_SHARED_LIBS=ON
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install -j
  STAMP_DIR ${GFLAGS_ROOT_DIR}/cmake-stamps
)
### jsoncpp
set(JSONCPP_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/jsoncpp)
ExternalProject_Add(
  jsoncpp_project
  GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
  GIT_TAG 1.9.5
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/jsoncpp-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/jsoncpp-build
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND cmake -S <SOURCE_DIR> -B <BINARY_DIR> -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${JSONCPP_ROOT_DIR} -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install && cp ${JSONCPP_ROOT_DIR}/lib64/cmake/jsoncpp/jsoncppConfig.cmake ${JSONCPP_ROOT_DIR}/lib64/cmake/jsoncpp/JsonCppConfig.cmake
  STAMP_DIR ${JSONCPP_ROOT_DIR}/cmake-stamps
)
### yaml-cpp
set(YAMLCPP_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/yaml-cpp)
ExternalProject_Add(
  yaml_cpp_project
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/yaml-cpp-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/yaml-cpp-build
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND cmake -S <SOURCE_DIR> -B <BINARY_DIR> -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${YAMLCPP_ROOT_DIR} -DBUILD_SHARED_LIBS=ON
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install -j
  STAMP_DIR ${YAMLCPP_ROOT_DIR}/cmake-stamps
)
set(YALANTINGLIBS_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/yalantinglibs)
### yalantinglibs
ExternalProject_Add(
  yalantinglibs_project
  GIT_REPOSITORY https://github.com/alibaba/yalantinglibs.git
  GIT_TAG 0.5.6
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/yalantinglibs-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/yalantinglibs-build
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND
    cmake
      -S <SOURCE_DIR>
      -B <BINARY_DIR>
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${YALANTINGLIBS_ROOT_DIR}
      -DBUILD_SHARED_LIBS=ON
      -DBUILD_EXAMPLES=OFF
      -DBUILD_BENCHMARK=OFF
      -DBUILD_UNIT_TESTS=OFF
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install -j
  STAMP_DIR ${YALANTINGLIBS_ROOT_DIR}/cmake-stamps
)
### openssl
set(OPENSSL_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/openssl)
ExternalProject_Add(
  openssl_project
  GIT_REPOSITORY https://github.com/openssl/openssl.git
  GIT_TAG ed00444368fcfe4fa74c80e553d1ab7e71cb2c9d
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/openssl-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/openssl-build
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND <SOURCE_DIR>/config --prefix=${OPENSSL_ROOT_DIR}
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install
  STAMP_DIR ${OPENSSL_ROOT_DIR}/cmake-stamps
)
### curl
set(CURL_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/curl)
ExternalProject_Add(
  curl_project
  DEPENDS openssl_project
  GIT_REPOSITORY https://github.com/curl/curl.git
  GIT_TAG c3add7130d7d81add205edbbb75fdfd1f38b3c68
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/curl-src
  BUILD_IN_SOURCE 1
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND
    cmake
      -S <SOURCE_DIR>
      -B <BINARY_DIR>
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${CURL_ROOT_DIR}
      -DBUILD_SHARED_LIBS=ON
      -DOPENSSL_ROOT_DIR=${OPENSSL_ROOT_DIR}
      -DCMAKE_INSTALL_RPATH=${OPENSSL_ROOT_DIR}/lib64
  BUILD_COMMAND make -j
  INSTALL_COMMAND make install
  STAMP_DIR ${CURL_ROOT_DIR}/cmake-stamps
)
### Mooncake
set(MOONCAKE_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/mooncake)
ExternalProject_Add(
  mooncake_project
  DEPENDS glog_project gflags_project jsoncpp_project yaml_cpp_project curl_project yalantinglibs_project
  GIT_REPOSITORY https://github.com/kvcache-ai/Mooncake.git
  GIT_TAG v0.3.7.post2
  SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/mooncake-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/mooncake-build
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND
    export CMAKE_PREFIX_PATH=${YAMLCPP_ROOT_DIR}:${YALANTINGLIBS_ROOT_DIR}:${CURL_ROOT_DIR}:${JSONCPP_ROOT_DIR} &&
    cmake
      -S <SOURCE_DIR>
      -B <BINARY_DIR>
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${MOONCAKE_ROOT_DIR}
      -DWITH_STORE=OFF
      -DBUILD_UNIT_TESTS=OFF
      -DBUILD_EXAMPLES=OFF
  BUILD_COMMAND
    export CPATH=${JSONCPP_ROOT_DIR}/include:${GFLAGS_ROOT_DIR}/include:${GLOG_ROOT_DIR}/include &&
    cmake --build <BINARY_DIR> -j
  INSTALL_COMMAND
    cp -r <SOURCE_DIR>/mooncake-transfer-engine/include ${MOONCAKE_ROOT_DIR} &&
    mkdir -p ${MOONCAKE_ROOT_DIR}/lib &&
    cp <BINARY_DIR>/mooncake-transfer-engine/src/common/base/libbase.a ${MOONCAKE_ROOT_DIR}/lib &&
    cp <BINARY_DIR>/mooncake-transfer-engine/src/libtransfer_engine.a ${MOONCAKE_ROOT_DIR}/lib
  STAMP_DIR ${MOONCAKE_ROOT_DIR}/cmake-stamps
)
# ### mooncake::libs
# find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
# add_library(mooncake_engine SHARED IMPORTED)
# add_dependencies(mooncake_engine glog_project jsoncpp_project curl_project)
# set_target_properties(
#   mooncake_engine
#   PROPERTIES
#   IMPORTED_LOCATION ${Python_SITELIB}/mooncake/engine.so
#   INTERFACE_INCLUDE_DIRECTORIES "${MOONCAKE_ROOT_DIR}/mooncake-transfer-engine/include;${GLOG_ROOT_DIR}/include;${JSONCPP_ROOT_DIR}/include"
#   INTERFACE_LINK_LIBRARIES "${CURL_ROOT_DIR}/lib/libcurl.so.4;${OPENSSL_ROOT_DIR}/lib64/libssl.so.4;${Python_SITELIB}/mooncake_transfer_engine.libs/libglog-c040a6f3.so.0.4.0;${Python_SITELIB}/mooncake_transfer_engine.libs/libgflags-f3c7dc50.so.2.2.2"
# )

add_library(mooncake INTERFACE)
add_dependencies(mooncake mooncake_project)
set_target_properties(
  mooncake
  PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${MOONCAKE_ROOT_DIR}/include;${GLOG_ROOT_DIR}/include;${GFLAGS_ROOT_DIR}/include;${JSONCPP_ROOT_DIR}/include;${YAMLCPP_ROOT_DIR}/include;${CURL_ROOT_DIR}/include"
  INTERFACE_LINK_LIBRARIES "${MOONCAKE_ROOT_DIR}/lib/libtransfer_engine.a;${MOONCAKE_ROOT_DIR}/lib/libbase.a;${CURL_ROOT_DIR}/lib64/libcurl.so;${GLOG_ROOT_DIR}/lib64/libglog.so;${GFLAGS_ROOT_DIR}/lib/libgflags.so;${JSONCPP_ROOT_DIR}/lib64/libjsoncpp.so;${YAMLCPP_ROOT_DIR}/lib64/libyaml-cpp.so;ibverbs;numa"
)

add_executable(httpstore httpstore.cpp)
add_dependencies(httpstore cpp_httplib_project)
target_include_directories(httpstore PUBLIC ${CPPHTTPLIB_ROOT_DIR})

add_executable(example example.cpp)
target_link_libraries(example PUBLIC mooncake)
target_include_directories(example PUBLIC ${CPPHTTPLIB_ROOT_DIR} ${MOONCAKE_ROOT_DIR}/include)
