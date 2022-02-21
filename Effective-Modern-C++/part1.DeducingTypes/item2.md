# Understand auto type deduction

该条款讲的是如何在含 ``auto`` 的表达式中确定变量的类型。此处所用的规则和条款一中的推导模板函数形参类型所用的规则一致。

注意：当 auto 修饰的变量用大括号语法 ``{...}`` 来初始化时，类型会被确定为 ``std::initializer_list`` 。

参考《Effective Modern C++》对应章节。
