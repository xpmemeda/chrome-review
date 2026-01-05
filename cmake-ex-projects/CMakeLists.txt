cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)

include(ExternalProject)
### boost-1.86.0
set(BOOST_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/boost-1.86.0)
ExternalProject_Add(
    boost_project
    URL https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-b2-nodocs.tar.gz
    SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/boost-src
    BUILD_IN_SOURCE 1
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND <SOURCE_DIR>/bootstrap.sh --prefix=${BOOST_ROOT_DIR}
    BUILD_COMMAND
        <SOURCE_DIR>/b2 install
            variant=release
            link=shared
            threading=multi
            runtime-link=shared
            cxxflags=-fPIC
            define=BOOST_STACKTRACE_USE_ADDR2LINE
            --with-stacktrace
            --with-system
            --with-filesystem
            -j32
    INSTALL_COMMAND ""
    STAMP_DIR ${BOOST_ROOT_DIR}
)
### hdf5-1.14.6
set(HDF5_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/hdf5-1.14.6)
ExternalProject_Add(
    hdf5_project
    GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
    GIT_TAG hdf5_1.14.6
    SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/hdf5-src
    BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/hdf5-build
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND
        cmake
            -S <SOURCE_DIR>
            -B <BINARY_DIR>
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX=${HDF5_ROOT_DIR}
    BUILD_COMMAND make -j
    INSTALL_COMMAND make install
    STAMP_DIR ${HDF5_ROOT_DIR}
)
# highfive-3.3.0
set(HIGHFIVE_ROOT_DIR ${CMAKE_BINARY_DIR}/3rds/highfive-3.3.0)
ExternalProject_Add(
    highfive_project
    DEPENDS hdf5_project
    GIT_REPOSITORY https://github.com/highfive-devs/highfive.git
    GIT_TAG v3.3.0
    SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/highfive-src
    BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/highfive-build
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND
        CMAKE_PREFIX_PATH=${HDF5_ROOT_DIR}
        cmake
            -S <SOURCE_DIR>
            -B <BINARY_DIR>
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX=${HIGHFIVE_ROOT_DIR}
    BUILD_COMMAND make -j
    INSTALL_COMMAND make install
    STAMP_DIR ${HIGHFIVE_ROOT_DIR}
)

set(BOOST_EXAMPLE boost-example)
add_executable(${BOOST_EXAMPLE} boost-example.cpp)
add_dependencies(${BOOST_EXAMPLE} boost_project)
target_compile_definitions(${BOOST_EXAMPLE} PRIVATE BOOST_STACKTRACE_USE_ADDR2LINE)
target_include_directories(${BOOST_EXAMPLE} PRIVATE ${BOOST_ROOT_DIR}/include)
target_link_directories(${BOOST_EXAMPLE} PRIVATE ${BOOST_ROOT_DIR}/lib)
target_link_libraries(${BOOST_EXAMPLE} PRIVATE boost_stacktrace_addr2line)

set(H5_EXAMPLE h5-example)
add_executable(${H5_EXAMPLE} h5-example.cpp)
add_dependencies(${H5_EXAMPLE} highfive_project)
target_include_directories(${H5_EXAMPLE} PRIVATE ${HIGHFIVE_ROOT_DIR}/include ${HDF5_ROOT_DIR}/include)
target_link_directories(${H5_EXAMPLE} PRIVATE ${HDF5_ROOT_DIR}/lib)
target_link_libraries(${H5_EXAMPLE} PRIVATE hdf5)
