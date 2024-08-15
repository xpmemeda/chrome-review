#pragma once

#include "torch/extension.h"

namespace cr {

void copy_v1(torch::Tensor& x, torch::Tensor& y);
void copy_v2(torch::Tensor& x, torch::Tensor& y);

}  // namespace cr
