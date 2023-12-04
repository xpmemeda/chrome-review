#pragma once

#include <vector>

namespace cr {

template <typename T>
void print_vec(const char* prefix, const std::vector<T>& vec) {
  std::cout << prefix;
  if (vec.empty()) {
    std::cout << "[]" << std::endl;
    return;
  }

  std::cout << "[";
  for (size_t i = 0; i < vec.size() - 1; ++i) {
    std::cout << vec[i] << ",";
  }
  std::cout << vec.back() << "]" << std::endl;
}

}  // namespace cr
