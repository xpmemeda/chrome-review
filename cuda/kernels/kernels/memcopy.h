#pragma once

#include "torch/extension.h"

namespace cr {

void memcopy(torch::Tensor& a, torch::Tensor& b);

}  // namespace cr
