#include <array>
#include <iostream>

extern int sum_array(const std::array<int, 4> &);

int main() {
  std::array<int, 4> arr = {1, 2, 3};
  int sum = sum_array(arr);
  std::cout << "sum = " << sum << std::endl;
  return 0;
}