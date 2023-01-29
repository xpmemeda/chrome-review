#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

template <typename T> std::vector<T> ones(size_t num_elem);

int main() {
  auto ones_vector = ones<int>(100);
  printf("ones_vector: %i\n", ones_vector[0]);
  return 0;
}