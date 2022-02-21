#include <iostream>
#include <string>
#include <fmt/core.h>

int main() {
    fmt::print("Hello world\n");
    {
        std::string str = fmt::format("The answer is {}", 2);
        std::cout << str << std::endl;
    }
    {
        std::string s = fmt::format("I'd rather be {1} than {0}.", "right", "happy");
        std::cout << s << std::endl;
    }
    return 0;
}