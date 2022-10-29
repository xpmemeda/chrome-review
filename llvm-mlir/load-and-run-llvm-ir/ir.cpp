#include <iostream>

extern "C" {

int func() {
  std::cout << "hello-world" << std::endl;
  return 0;
}
}