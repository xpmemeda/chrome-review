### C++ auto 类型推断规则

```cpp
#include <type_traits>

int main() {
  {
    const int x = 0; auto y = x;
    // ``auto`` will not infer the cv type qualifiers
    static_assert(std::is_same_v<decltype(y), int>);
  }
  {
    const int x = 0; auto& y = x;
    // ``auto&`` will infer the cv type qualifiers
    static_assert(std::is_same_v<decltype(y), const int&>);
    // error, ``auto&`` cannot be used to represent rvalue reference.
    // auto& y = 0;
  }
  {
    const int x = 0; const auto& y = x;
    static_assert(std::is_same_v<decltype(y), const int&>);
    const auto& z = 0;
    static_assert(std::is_same_v<decltype(z), const int&>);
  }
  {
    // ``auto&&`` is universal reference
    const int x = 0; auto&& y = x;
    static_assert(std::is_same_v<decltype(y), const int&>);
    auto&& z = 0; z = 1;
    static_assert(std::is_same_v<decltype(z), int&&>);
    // error, ``const auto&&`` is const rvalue reference.
    // const auto&& zz = x;
  }
  return 0;
}
```