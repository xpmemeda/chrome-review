#pragma once

#include "torch/extension.h"

namespace cr {

void softmax_v1(torch::Tensor& x, torch::Tensor& out);
void softmax_v2(torch::Tensor& x, torch::Tensor& out);
void softmax_v3(torch::Tensor& x, torch::Tensor& out);
void softmax_v4(torch::Tensor& x, torch::Tensor& out);

}  // namespace cr
