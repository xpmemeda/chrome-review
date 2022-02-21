#include <iostream>
#include <utility>

class A {
public:
  bool fun();
};

class B {};

namespace {
template <class T, class = std::void_t<>> struct HasFun : std::false_type {};
template <class T>
struct HasFun<T, std::void_t<decltype(std::declval<T>().fun())>>
    : std::true_type {};
} // namespace

int main() {
  std::cout << HasFun<A>::value << std::endl;
  std::cout << HasFun<B>::value << std::endl;
  return 0;
}
