# Understand decltype

该条款讲的是 ``decltype`` 的推理规则：
表达式的结果是什么类型，``decltype`` 就会推导出什么类型，保留 ``cv`` 类型符号，保留引用。

需要注意的是：

```cpp
// 复杂左值表达式：单变量名 + 圆括号 --> int&
int x = 0; decltype((x)) y = x;
// x + y 可以视为 ::operator+() 函数的返回值，所以 decltype 推导类型的还是 int
int x, y; decltype((x + y)) z = x;
```

参考《Effective Modern C++》对应章节。
