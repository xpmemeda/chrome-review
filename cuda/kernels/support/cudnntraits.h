#pragma once

#include "cuda_fp16.h"
#include "cudnn.h"

namespace cr {

template <typename T>
struct CudnnDataTypeTrait;

template <>
struct CudnnDataTypeTrait<float> {
  static constexpr cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
};

template <>
struct CudnnDataTypeTrait<half> {
  static constexpr cudnnDataType_t data_type = CUDNN_DATA_HALF;
};

}  // namespace cr
