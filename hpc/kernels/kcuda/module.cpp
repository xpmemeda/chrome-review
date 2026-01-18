#include <functional>
#include <mutex>
#include <unordered_map>

#include "pybind11/pybind11.h"

namespace cr {

using RegF = std::function<void(pybind11::module&)>;

std::vector<std::function<void(pybind11::module&)>>& registers() {
  static std::vector<std::function<void(pybind11::module&)>> registers;
  return registers;
}

static std::mutex mu;

void add_register(std::function<void(pybind11::module&)> fn) {
  std::lock_guard<std::mutex> _(mu);
  registers().push_back(fn);
}

void register_fn(pybind11::module& m) {
  std::lock_guard<std::mutex> _(mu);
  for (auto& fn : registers()) {
    fn(m);
  }
}

PYBIND11_MODULE(kernels, m) {
  m.attr("__name__") = "kernels";
  m.attr("__package__") = m.attr("__name__");

  pybind11::module::import("sys").attr("modules")[m.attr("__name__")] = m;
  pybind11::module::import("torch");

  register_fn(m);
}

}  // namespace cr
