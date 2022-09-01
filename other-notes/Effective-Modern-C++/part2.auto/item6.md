# Use the explicitly typed initializer idiom when auto deduces undesired types

该条款讲了两点：

1. ``auto`` 关键字和代理类天生是不兼容的，后者在大部分情况下被设计为仅仅存活在单个语句之内，而 ``auto`` 会打乱这种设计，从而出现未知的问题。

>> ``std::vector<bool>`` 使用一种压缩的比特形式表示其持有的元素，比特无法直接取引用，所以 ``operator[]`` 必须返回一个代理类来体现其引用特征。

2. 在涉及到类型转换时，应该使用 ``static_cast`` 把程序员这种“故意的行为”表现出来。

>> ```cpp
>> double calc();
>> auto r = static_cast<float>(calc());
>> ```

参考《Effective Modern C++》对应章节。
