# OOM

- total-vm: 虚拟地址空间大小
- anon-rss: 匿名内存（heap / stack / malloc）
- file-rss: 文件映射（mmap 文件、so）
- shmem-rss: 共享内存（tmpfs / shm / CUDA / torch / IPC）
- pgtables: 页表占用的内存大小。
- oom_score_adj: OOM 权重，区间 [-1000 ~ +1000]。-1000：几乎不可杀，+1000：优先杀


dmesg 是 Linux 里一个非常重要、也非常常用的系统诊断工具，主要作用是：

✅ 查看内核（kernel）日志消息——也就是操作系统“内核层”发生的所有重要事件记录。

一、dmesg 到底在看什么？

dmesg 读取的是 内核环形缓冲区（kernel ring buffer），里面记录的都是：

硬件相关事件（磁盘、网卡、GPU、USB、内存）

驱动加载与错误

内存不足（OOM）

I/O 错误

内核异常、崩溃、死锁

启动过程的关键信息

它看的 不是普通应用日志，而是 内核级别的“底层真相”。

二、它常用吗？一句话结论 ✅

✅ 非常常用

只要你遇到下面这些问题，dmesg 几乎是“第一现场”：

程序无缘无故被杀掉

I/O 异常

显卡 / CUDA 出问题

网卡 or 网速异常

磁盘挂载失败

内存不足

Docker / 容器 mysterious 挂掉

在你这种 C++ / CUDA / 容器 / cgroup / pinned memory / OOM 场景下，dmesg 基本是必查工具 🔥。