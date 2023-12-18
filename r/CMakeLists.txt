cmake_minimum_required(VERSION 3.20)

project(r)

macro(list_sub_directories RESULT CURDIR)
  file(GLOB children RELATIVE ${CURDIR} ${CURDIR}/r*)
  set(DIRLIST "")
  foreach(child ${children})
    if(IS_DIRECTORY ${CURDIR}/${child})
      list(APPEND DIRLIST ${child})
    endif()
  endforeach()
  set(${RESULT} ${DIRLIST})
endmacro()

list_sub_directories(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})
foreach(SUBDIR ${SUBDIRS})
  add_subdirectory(${SUBDIR})
endforeach(SUBDIR ${SUBDIRS})
