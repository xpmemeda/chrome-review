#pragma once

#include "torch/extension.h"

namespace cr {

void gemm_v1(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c);
void gemm_v2(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c);

}  // namespace cr
