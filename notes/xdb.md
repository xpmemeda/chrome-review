# Python:Pdb

## 如何调试不能通过命令行交互的 Python 代码，比如子进程？

**使用 rpdb 重定向 Pdb 的交互 I/O**

Step1: 修改源码

```python
if not hasattr(self, 'rpdb'):
    from remote_pdb import RemotePdb
    self.rpdb = RemotePdb("127.0.0.1", 4444)
self.rpdb.set_trace()
```

Step2: 连接端口

```bash
nc 127.0.0.1 4444
```


# Cxx:Gdb

## 捕获代码中被 ``try-catch`` 的异常

```bash
(gdb) catch throw c10::Error
```

## 打印所有线程的栈

```bash
(gdb) thread apply all bt 5
```

## 查看和修改系统当前的 coredump 文件路径

- %e: 可执行文件名
- %p: PID
- %u: UID
- %g: GID
- %t: 时间戳
- %h: 主机名
- %s: 信号

```bash
sysctl kernel.core_pattern
```

情况一：

输出一个路径，比如 ```/tmp/core.%e.%p.%t```。


情况二：

输出类似 ```|/usr/lib/systemd/systemd-coredump %P %u %g %s %t %c %h %e```，那么当前系统的 coredump 已经被 systemd:coredumpctl 接管。

查看 systemd 保存的 core
```bash
coredumpctl list
coredumpctl into $PID
```
