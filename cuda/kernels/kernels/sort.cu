#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <iostream>

#include "c10/cuda/CUDAStream.h"
#include "torch/all.h"

#include "../module.h"
#include "../support/assert.h"

namespace {

void sort_1(torch::Tensor& r, torch::Tensor& x) {
  r.copy_(x);
  thrust::device_ptr<half> devptr(reinterpret_cast<half*>(r.data_ptr()));

  const int num_tokens = r.size(0);
  const int vocab_size = r.size(1);
  for (int i = 0; i < num_tokens; ++i) {
    thrust::sort(devptr + i * vocab_size, devptr + (i + 1) * vocab_size);
  }
}

static cr::Register _([](pybind11::module& m) { m.def("sort_1", &sort_1); });

}  // namespace