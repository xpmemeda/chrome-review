#include <functional>
#include <mutex>
#include <unordered_map>

#include "pybind11/pybind11.h"
#include "torch/all.h"

#include "kernels/conv.h"
#include "kernels/gemm.h"
#include "kernels/softmax.h"
#include "module.h"

namespace cr {

std::vector<std::function<void(pybind11::module&)>>& get_registers() {
  static std::vector<std::function<void(pybind11::module&)>> registers;
  return registers;
}
static std::mutex mu;
void add_register(std::function<void(pybind11::module&)> fn) {
  std::lock_guard<std::mutex> _(mu);
  get_registers().push_back(fn);
}

void register_fn(pybind11::module& m) {
  std::lock_guard<std::mutex> _(mu);
  for (auto& fn : get_registers()) {
    fn(m);
  }
}

PYBIND11_MODULE(kernels, m) {
  m.attr("__name__") = "kernels";
  m.attr("__package__") = m.attr("__name__");

  pybind11::module::import("sys").attr("modules")[m.attr("__name__")] = m;
  pybind11::module::import("torch");

  m.def("softmax_v1", &softmax_v1);
  m.def("softmax_v2", &softmax_v2);
  m.def("softmax_v3", &softmax_v3);
  m.def("softmax_v4", &softmax_v4);

  m.def("gemm_v1", &gemm_v1);
  m.def("gemm_v2", &gemm_v2);

  m.def("conv_v1", &conv_v1);

  register_fn(m);
}

}  // namespace cr
