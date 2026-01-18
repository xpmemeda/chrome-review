#pragma once

#include <functional>

#include "pybind11/pybind11.h"

// NOTE: Include the header file to recognize torch::Tensor type
#include "torch/csrc/autograd/python_variable.h"

namespace cr {

void add_register(std::function<void(pybind11::module&)> fn);

class Register {
 public:
  using RegF = std::function<void(pybind11::module&)>;

  Register(RegF fn) { add_register(fn); }
};

}  // namespace cr
