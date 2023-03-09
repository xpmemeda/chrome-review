#include <array>

int sum_array(const std::array<int, 4> &a) {
  int sum = 0;
  for (auto x : a)
    sum += x;
  return sum;
}