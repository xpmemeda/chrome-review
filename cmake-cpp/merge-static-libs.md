### 把第三方静态库直接嵌入到自己的静态库里面

**动机**

通常``target_link_libraries``只会在``mylib-cmake-config.cmake``文件里面声明一项依赖，如果想要把自己的库提供给其他人，仍然要提供第三方库文件，通常是``3rdlib.a``，不太方便。

**解决办法**

用下面这段cmake语句，把``3rdlib.a``里面的所有Object文件拷贝到``mylib.a``。

```cmake
set(LIB_PATH path/to/3rdlib.a)
set(OBJ_DIR tmp/dir/to/save/object/files)
file(MAKE_DIRECTORY ${OBJ_DIR})
execute_process(COMMAND ${CMAKE_AR} -x ${LIB_PATH} WORKING_DIRECTORY ${OBJ_DIR})
file(GLOB OBJ_FILES ${OBJ_DIR}/*.o)
foreach(OBJ_FILE ${OBJ_FILES})
    target_sources(mylib PRIVATE ${OBJ_FILE})
endforeach()
```