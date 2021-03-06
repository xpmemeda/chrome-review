### 迭代器（Iterator）和可迭代对象（Iterable）有什么区别？

前者定义了 ``__iter__`` 和 ``__next__`` 这两个方法，``__iter__`` 返回自身，``__next__`` 返回迭代时的产出对象。
后者只需要定义 ``__iter__``，通过该方法返回一个迭代器。

### 为什么迭代器要实现 ``__iter__`` 方法？

在 python 的 ``for ... in ...`` 语法中，需要在后面跟一个可迭代对象（Iterable），可迭代对象通过 ``__iter__`` 方法来生成一个迭代器以供循环语句使用。
所以为了让迭代器也可以跟在 ``for ... in ...`` 语句后面，一般会实现 ``__iter__`` 方法，且直接返回自身就可以。

### 迭代器和生成器（Generator）有何不同？

所有的生成器都是迭代器，但是构造方法更简单，只需要定义包含一个或多个 ``yield`` 语句的函数即可，无需创建类并实现 ``__iter__`` 和 ``__next__`` 方法。
当包含 ``yield`` 语句的函数执行完毕时，该生成器会报一个类型为 ``StopIteration`` 的错误。
