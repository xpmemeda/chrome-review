#include <iostream>

#include "torch/csrc/api/include/torch/all.h"

#include "templates/math.cuh"
#include "templates/print.cuh"

__global__ void print(void* p) {
  auto v = cr::F16Operator::sum(static_cast<uint2*>(p)[0]);
  printf("%f\n", v);
}

int main() {
  auto x = torch::ones({4}, torch::TensorOptions(torch::kCUDA, 0).dtype(torch::ScalarType::Half));
  std::cout << x << std::endl;
  print<<<1, 1>>>(x.data_ptr());
  return 0;
}