cmake_minimum_required(VERSION 3.20)

project(hello-world)


find_package(CUDAToolkit REQUIRED)

# 获取所有变量的列表
get_cmake_property(all_vars VARIABLES)

# 打印所有变量及其值
foreach(var ${all_vars})
    message(STATUS "${var} = ${${var}}")
endforeach()
