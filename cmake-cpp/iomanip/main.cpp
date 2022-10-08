#include <iomanip>
#include <iostream>
#include <string>

int main() {
  std::cout << std::setfill('*') << std::left << std::setw(10) << "hello"
            << std::setw(10) << "world" << 0 << std::endl;
  std::cout << std::setprecision(3) << std::setw(10) << 1.1f << 1.11111f
            << std::endl;
  return 0;
}