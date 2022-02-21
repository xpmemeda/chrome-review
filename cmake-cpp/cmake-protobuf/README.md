### 在 cmake 项目中使用 protobuf

**安装 protobuf**

从 github 源码仓库安装，下载 release 包，以版本 3.18.0 为例：

```bash
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.18.0/protobuf-all-3.18.0.zip
unzip protobuf-all-3.18.0.zip && cd protobuf-3.18.0
./configure -prefix=/usr/local/
sudo make -j && sudo make install
```

**使用 cmake 编译使用 protobuf 的程序**

见 ``CmakeLists.txt``。

参考1：https://cmake.org/cmake/help/latest/module/FindProtobuf.html

参考2：https://developers.google.com/protocol-buffers/docs/cpptutorial
