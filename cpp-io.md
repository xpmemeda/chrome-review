### 读取全部内容

```cpp
#include <fstream>
#include <string>

int main() {
    using std::ios;
    std::ifstream infile("cpp-io.md", ios::binary);
    if (!infile) {
        std::cout << "Open file failed" << std::endl;
        return -1;
    }
    std::string data(std::istreambuf_iterator<char>(infile), {});
    std::cout << data << std::endl;
    return 0;
}
```


### 分段读取文件

```cpp
#include <iostream>
#include <fstream>

int main() {
    constexpr size_t bs = 10;
    char buffer[bs];
    using std::ios;
    std::ifstream infile("cpp-io.md", ios::binary | ios::ate);
    if (!infile) {
        std::cout << "Open file failed" << std::endl;
        return -1;
    }
    std::cout << "File length: " << infile.tellg() << std::endl;
    infile.seekg();
    for (size_t i = 0; i < 10; ++i) {
        infile.read(buffer, bs);
        std::cout << "Because there may be a value of 0 in buffer, "
                    "it may not print the entire content" << std::endl;
        std::cout << buffer << std::endl;
    }
    return 0;
}
```
