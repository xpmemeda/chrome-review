#include "./tensor.h"

#include "torch/csrc/api/include/torch/all.h"

namespace cr {

Tensor Tensor::referenceFromTorchTensor(const torch::Tensor& x) {
  void* buffer = reinterpret_cast<void*>(x.data_ptr());
  std::vector<int64_t> sizes(x.sizes().begin(), x.sizes().end());
  std::vector<int64_t> strides(x.strides().begin(), x.strides().end());
  return Tensor(buffer, std::move(sizes), std::move(strides));
}

}  // namespace cr
