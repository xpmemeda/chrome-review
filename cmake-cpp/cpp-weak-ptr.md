### C++ weak_ptr 的使用场景

**避免循环引用**

```cpp
#include <memory>
#include <iostream>

struct ListNode {
    ListNode(int _val) : val(_val) {
        std::cout << "Ctor" << std::endl;
    }
    ~ListNode() {
        std::cout << "Dtor" << std::endl;
    }
    int val;
    std::shared_ptr<ListNode> left;
    std::shared_ptr<ListNode> rigint;
};

/*
 * Ctor
 * Ctor
 * 2
 * 2
 */
int main() {
    auto n1 = std::make_shared<ListNode>(1);
    auto n2 = std::make_shared<ListNode>(2);
    n1->rigint = n2;
    n2->left = n1;
    std::cout << n1.use_count() << std::endl;
    std::cout << n2.use_count() << std::endl;
    return 0;
}
```

示例程序会导致 ``ListNode`` 之间的循环引用，使得 shared_ptr 的引用计数不被清零，最终引发内存泄漏。

使用 weak_ptr 可以解决这个问题：

```cpp
#include <memory>
#include <iostream>

struct ListNode {
    ListNode(int _val) : val(_val) {
        std::cout << "Ctor" << std::endl;
    }
    ~ListNode() {
        std::cout << "Dtor" << std::endl;
    }
    int val;
    std::shared_ptr<ListNode> left;
    std::weak_ptr<ListNode> rigint;
};

/*
 * Ctor
 * Ctor
 * 2
 * 1
 * Dtor
 * Dtor
 */
int main() {
    auto n1 = std::make_shared<ListNode>(1);
    auto n2 = std::make_shared<ListNode>(2);
    n1->rigint = n2;
    n2->left = n1;
    std::cout << n1.use_count() << std::endl;
    std::cout << n2.use_count() << std::endl;
    return 0;
}
```


**实现观察者模式**


假设需要一个集合来记录所有存活的 shared_ptr，那么就只能 weak_ptr 来实现。理由是如果集合中使用 shared_ptr 会导致被记录的对象引用计数不被清零，永远都不被释放，而使用裸指针则没法判断该指针是否悬空。

> :warning: 用 weak_ptr 实现缓存也可以归于此类。
