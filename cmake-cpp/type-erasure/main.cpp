#include "type.h"

#include <iostream>

int main() {
  class B {
   public:
    void fn1(int x) { std::cout << "int: " << x << std::endl; }
    void fn2(float x) { std::cout << "float: " << x << std::endl; }
  };
  fn(B());
  return 0;
};