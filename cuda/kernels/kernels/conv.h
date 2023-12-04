#pragma once

#include <array>

#include "torch/extension.h"

namespace cr {

void conv_v1(torch::Tensor& input, torch::Tensor& kernel, torch::Tensor& output, int group,
    const std::vector<int>& paddings, const std::vector<int>& strides, const std::vector<int>& dilates);

}
