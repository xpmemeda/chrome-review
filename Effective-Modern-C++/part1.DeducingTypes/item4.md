# Know how to view deduced types

该条款介绍如何查看 ``模板``，``auto`` 或者是 ``decltype`` 推导出来的实际类型：

1. 依靠 IDE。
2. 在编译器故意写 bug 引发报错信息。
3. 运行时打印。

提到运行时打印类型，首先就会想到 ``typeid``，但是它并不可靠，不仅输出的类型名难以解读，甚至有时会给出错误的结果。
建议使用下面这种方式：

```cpp
#include <boost/type_index.hpp>

template <typename T>
void fun(const T& param);

{
    using boost::typeindex::type_id_with_cvr;
    std::cout << "T = " << type_id_with_cvr<T>().pretty_name() << std::endl;
    std::cout << "param = " << type_id_with_cvr<decltype(param)>().pretty_name() << std::endl;
}
```

参考《Effective Modern C++》对应章节。
