## 安装 gperftools

从github下载[源码](https://github.com/gperftools/gperftools)安装：

```bash
./autogen.sh
./configure
make -j8
sudo make install
```

## 编译、运行待测试程序

编译：要求链接tcmalloc

运行：
```bash
HEAPPROFILE=<path to log file> <path to binary> [binary args]
```

## 查看内存结果

通过log生成可读的pdf文件：

```bash
pprof --pdf <path to binary> <path to log file>.<index>.heap > <pdf path>
```
