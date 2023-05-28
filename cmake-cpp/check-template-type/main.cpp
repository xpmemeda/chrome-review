#include <iostream>
#include <string>
#include <type_traits>

template <class T> std::string type_name() {
  using TR = typename std::remove_reference_t<T>;

  std::string r = "";
  if (std::is_const_v<TR>)
    r = std::string("const ") + r;
  if (std::is_volatile_v<TR>)
    r = std::string("volatile ") + r;

  if (std::is_lvalue_reference_v<T>)
    r += std::string("&");
  else if (std::is_rvalue_reference_v<T>)
    r += std::string("&&");

  return r;
}

int main() {
  std::cout << type_name<int &>() << std::endl;
  return 0;
}