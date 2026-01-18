#include <ATen/Tensor.h>
#include <cuda.h>
#include <cudaProfiler.h>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <vector>

#include "./module.h"

namespace {

constexpr int BLOCK_SIZE = 128;
constexpr int REPEAT = 4096;

template <int STRIDE>
__global__ void kernel_bankconflict(float* out) {
  __shared__ volatile float s[BLOCK_SIZE][STRIDE];
  int tid = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;

  s[tid][0] = float(tid);
  __syncthreads();

  float acc = 0.f;
  for (int it = 0; it < REPEAT; ++it) {
    acc += s[tid][0];
    s[tid][0] = acc;
  }
  out[gtid] = acc;
}

void benchmark_bankconflict(at::Tensor& x, int stride) {
  int grid_size = (x.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE;

  if (stride == 1) {
    return kernel_bankconflict<1><<<grid_size, BLOCK_SIZE>>>(x.data_ptr<float>());
  }
  if (stride == 2) {
    return kernel_bankconflict<2><<<grid_size, BLOCK_SIZE>>>(x.data_ptr<float>());
  }
  if (stride == 16) {
    return kernel_bankconflict<16><<<grid_size, BLOCK_SIZE>>>(x.data_ptr<float>());
  }
  if (stride == 17) {
    return kernel_bankconflict<17><<<grid_size, BLOCK_SIZE>>>(x.data_ptr<float>());
  }

  throw std::runtime_error("unsupported stride.");
}

static cr::Register _([](pybind11::module& m) { m.def("benchmark_bankconflict", &benchmark_bankconflict); });

}  // namespace
