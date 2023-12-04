#pragma once

#include <cstdint>
#include <vector>

#include "torch/csrc/api/include/torch/all.h"

namespace cr {

class Tensor {
 public:
  Tensor() : Tensor(nullptr, {}, {}) {}
  Tensor(void* buffer, std::vector<int64_t> sizes, std::vector<int64_t> strides)
      : buffer_(buffer), sizes_(std::move(sizes)), strides_(strides) {}

  static Tensor referenceFromTorchTensor(const torch::Tensor&);

  template <typename T>
  T* data() {
    return reinterpret_cast<T*>(buffer_);
  }

  template <typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(buffer_);
  }

  int64_t size(size_t index) const { return sizes_.at(index); }
  int64_t stride(size_t index) const { return strides_.at(index); }

 private:
  void* buffer_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
};

}  // namespace cr
