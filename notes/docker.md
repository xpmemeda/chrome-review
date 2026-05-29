# 磁盘空间爆了怎么解决

## 查看当前的磁盘用量

1. 获取docker数据目录
```bash
cat /etc/docker/daemon.json | grep "data-root"
```

2. 获取每个目录的占用空间大小
```bash
du -sh ${data-root}/*
```
每个目录的意思是：

- containers：容器的数据和日志。
- image：镜像的元数据和层信息。
- overlay2：其他存储驱动目录：镜像和容器的实际文件系统层。
- volumes：Docker 卷的数据。
- network：Docker 网络的配置和元数据。
- plugins：Docker 插件的元数据和数据

大头是containers和overlay2，前者是运行中的容器占用的磁盘空间，后者是镜像占用的空间。

3. 获取哪个镜像或者容器占用磁盘最多
```bash
du -sh ${data-root}/containers/*
```
找到最多的那些干掉。

## 备注

使用docker自带的``docker system df``显示的磁盘数据信息并不管用，有些容器在运行中产生的很多日志or数据不会被docker统计到。


# 容器内

## 怎么看自己是不是在容器内

```bash
cat /proc/1/cgroup | grep kubepods
```

看看有没有内容，有的话则说明在容器内，且该容器是由 K8s 管理的。

## 怎么看可用的 CPU 核数

**1. 看 CFS quota**

```bash
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
```

二者相除，出现 -1 表示无 quota 限制。

**2. 看可调度的 CPU 核数**

```bash
nproc
```