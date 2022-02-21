# 第十一章 运行库

## 11.1 入口函数和程序初始化

### 11.1.1 程序从 main 开始吗

操作系统装载程序后，首先运行的代码并不是 main 的第一行，而是某些别的代码，这些代码负责准备好 main 函数执行所需要的环境，并且负责调用 main 函数。
这时候 main 函数里面才可以放心地写各种代码：申请内存、使用系统调用、触发异常、访问 IO。
在 main 返回之后，它会记录 main 函数的返回值，调用 atexit 注册的函数，然后结束进程。

一个经典的程序运行步骤如下：

1. 操作系统创建进程之后，把控制权交给程序的入口，这个入口往往是运行库中的某个入口函数。
2. 入口函数对运行库和程序运行环境进行初始化，包括堆、I/O、线程、全局变量构造等等。
3. 入口函数在完成初始化之后，调用 main 函数，正式开始执行程序主体部分。
4. main 函数执行完毕之后，返回到入口函数，入口函数进行清理工作，包括全局变量析构，堆销毁，关闭 I/O 等，然后进行系统调用结束进程。

### 11.1.2 入口函数如何实现

glibc 的程序入口函数为 ``_start``，该函数由汇编实现，且和平台相关。
i386 的实现如下：

```x86asm
_start:
    xor %ebp, %ebp    ; %ebp = 0
    pop %esi          ; %esi = argc
    mov %esp, %ecx    ; %ecx = argv + __environ

    push %esp
    push %edx
    push $__libc_csu_fini    ; init function called in __libc_start_main
    push $__libc_csu_init    ; fini function called in __libc_start_main
    push %ecx
    push %esi
    push main                ; main called in __libc_start_main
    call __libc_start_main
```

<div align="center">
<img src="https://github.com/xpmemeda/task/raw/master/chrome-review-list/Link-Load-and-Library/resource/11-1.jpg" width = "600" height = "200" alt="图 11-1 环境变量和参数数组"/>
</div>

上述实现运行时的栈区内存如图 11-1，``_start`` 函数完成了**堆栈的初始化和部分寄存器的赋值**，然后调用 ``__libc_start_main`` 来完成进一步的程序初始化。

``__libc_start_main`` 会进一步区分 ``argv`` 和 ``__environ``，**调用程序初始化函数，设置退出函数**，然后再调用 main 函数。
main 函数结束返回至入口函数后，入口函数会调用 ``exit`` 结束程序，而 ``exit`` 在执行退出系统调用之前，会先执行 ``atexit`` 预设的 fini 函数。即使程序员在代码中手动调 ``exit``，预设的 fini 函数还是会被执行。

《程序员的自我修养》此处对入口函数的功能解析并不完全，
但总之它要完成运行库和程序运行环境的初始化，然后调用 main 函数。

### 11.1.3 运行库与 I/O

### 11.1.4 MSVC CRT 的入口函数和初始化

## 11.2 C/C++ 运行库

### 11.2.1 C 语言运行库

任何一个 C 程序，它的背后都有一套庞大的代码来进行支撑，以使得该程序能够正常运行。
这套代码至少包括入口函数，及其所依赖的函数所构成的函数集合。
当然，它还理应包括各种标准库函数的实现。

这样的一个代码集合被称为运行时库（Runtime Library）。而 C 语言的运行时库被称为 C 运行库（CRT）。

一个 C 语言运行库大致包含了如下功能：

- 启动与退出：包括入口函数和入口函数所依赖的其他函数等。
- 标准函数：由 C 语言标准规定的 C 语言标准库所拥有的函数实现。
- I/O：I/O 功能的封装和实现。
- 堆：堆的封装和实现。
- 语言实现：语言中的一些特殊功能的实现。
- 调试：实现调试功能的代码。

### 11.2.2 C 语言标准库

C 语言在 AT&T 的贝尔实验室诞生，初生的 C 语言在功能上非常不完善，例如不提供 I/O 相关的函数。
上世纪 80 年代时，C 语言出现了大量的变种和多种不同的基础函数库，这对代码迁移等方面造成巨大的障碍。
于是美国国家标准协会（American National Standards Institute，ANSI）在 1983 年成立了一个委员会，旨在对 C 语言进行标准化，此委员会所建立的标准被称为 ANSI C。
第一个完整的 C 语言标准建立于 1989 年，此版本的 C 语言标准称为 C89。1995 年和 1999 年，ANSI C 标准库经过两次扩充后，其面貌一直延续至今。

ANSI C 标准库由 24 个 C 头文件组成，如

- 标准输入输出和文件操作（stdio.h）
- 字符串操作（string.h）
- 数学函数（math.h）
- 资源管理（stdlib.h）
- 时间和日期（time.h）
- 断言（assert.h）
- 变长参数（stdarg.h）
- 非局部跳转（setjmp.h）

### 11.2.3 glibc 与 MSVC CRT

Linux 和 Windows 平台的两个主要 C 语言运行库分别为 glibc（GNU C Library）和 MSVCRT（Microsoft Visual C Runtime）。

glibc 的发布版本主要由两部分组成，一部分是头文件，比如 stdio.h、stdlib.h 等，他们往往位于 /usr/include；
另一部分则是库的二进制文件部分，动态库位于 /lib/libc.so.6，静态库位于 /usr/lib/libc.a。

glibc 除了标准库，还有几个辅助程序运行的运行库，这几个文件才是真正的**运行库**，他们就是
/usr/lib/crt1.o、/usr/lib/crti.o 和 /usr/lib/crtn.o。

crt1.o 里面包含的就是入口函数 ``_start``，由他负责 ``__libc_start_main`` 初始化 libc 并且调用 ``main`` 函数进入真正的程序主体。crt1.o 的原名叫做 crt0.o，确保链接时它第一个输入的文件。

后来由于 C++ 的出现和 ELF 文件的改进，出现了必须在 ``main`` 之前执行的全局、静态对象的构造函数和在 ``main`` 函数之后执行的对应对象的析构函数。
为了满足类似需求，运行库在每个目标文件中引入了两个和初始化相关的段 ``.init`` 和 ``.finit``，运行库会保证所有位于这两个段中的代码会先/后于 ``main`` 函数执行。
链接器在进行链接时，会把所有输入文件的 ``.init`` 和 ``.finit`` 段按顺序收集起来，然后把他们拼成输出文件的 ``.init`` 和 ``.finit``。
但是这两个段中所包含的指令还需要一些辅助代码来帮助它们启动（比如计算 GOT 之类的），于是引入了两个目标文件 crti.o 和 crtn.o。

于此同时 crt0.o 也升级成了 crt1.o，在调用 ``__libc_start_main`` 是增加了两个额外参数 ``__libc_csu_init`` 和 ``__libc_csu_fini``，他们分别负责调用 ``_init()`` 和 ``_finit()``。

crti.o 和 crtn.o 实际包含的是 ``_init()`` 和 ``_finit()`` 函数的开头和结尾部分，当这两个文件和其他目标文件安装顺序连接起来之后，刚刚好就形成了两个完整的 ``_init()`` 和 ``_finit()``。

**GCC 平台相关目标文件**

在可执行文件中除了会包含 crt1.o、crti.o 和 crtn.o 之外，还会存在 crtbeginT.o、libgcc.a、libgcc_eh.a、crtend.o。
这些文件其实不属于 glibc，他们是 GCC 的一部分，都位于 GCC 的安装目录下。

Q：为什么可执行文件会包含这些 GCC 的目标文件？

A：glibc 不知道 C++ 具体的实现细节，要完成全局变量的构造和析构，必须由 GCC 来提供执行的二进制代码。

**C++ CRT**

如果程序是用 C++ 编写的，则需要额外链接 C++ 标准库。C++ 标准库包括 iostream、string、map 等。

Q：如果一个程序里面的不同共享库使用了不同的 CRT，会不会有问题？

A：情况比较复杂，有时可以有时不行，最好可以避免这种场景。

## 11.3 运行库与多线程

### 11.3.1 CRT 的多线程困扰

多线程程序并非所有数据都是共享的，如下表：

|线程私有|线程间共享|
|:-:|:-:|
|局部变量、函数的参数、TLS 数据、寄存器的值|全局变量、堆上的数据、静态变量、程序代码|

TLS：线程局部存储（Thread Local Storage）

现有的 C/C++ 标准中没有提到多线程，所以运行库也无能为力。不过 MSVC CRT 和 glibc 都提供了一些可选的线程库以供使用。
比如 glibc 的 pthread（POSIX Thread），它提供了 ``pthread_create``、``pthread_exit`` 等函数用于线程库的创建和退出。
很明显，这些函数都不属于标准的运行库，他们都是平台相关的。

CRT 早年间并没有考虑到多线程环境，因此吃了不少苦头，比如。。。

### 11.3.2 CRT 改进

- 使用 TLS：保证线程安全。

- 加锁：保证线程安全。

- 改进函数的调用方式：提供新的线程安全的函数接口。

### 11.3.3 线程局部存储实现

GCC 中，只需要在变量前加上关键字 ``__thread`` 即可声明线程局部存储：

```cpp
__thread int number;
```

glibc 具体实现方式：《程序员的自我修养》一书中未详细说明。

## 11.4 C++ 全局构造与析构

### 11.4.1 glibc 全局构造与析构

如前所述，程序入口 ``_start`` 会通过 ``__lib_start_main`` 调用 ``__libc_csu_init`` 函数，该函数中调用的就是 ``.init`` 段代码。
``.init`` 段中包含了可执行文件中所有全局变量的初始化。

```
_start -> __libc_start_main -> __libc_csu_init -> _init
```

### 11.4.2 MSVC CRT 的全局构造与析构

## 11.5 fread 实现
