#include <assert.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "custom_class.h"
#include "numpy_functions.h"

double add(double i, double j) { return i + j; }

std::vector<double> add_vector(const std::vector<double>& x, const std::vector<double>& y) {
  assert(x.size() == y.size());
  std::vector<double> z(x.size());
  for (int i = 0; i < x.size(); ++i) {
    z[i] = x[i] + y[i];
  }
  return z;
}
void printUtf8Bytes(const std::string& utf8Str) {
  std::cout << "b'";
  for (unsigned char c : utf8Str) {
    std::cout << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c);
  }
  std::cout << "'" << std::endl;
}

namespace py = pybind11;

py::module submodule(py::module parent, const char* name, const char* doc) {
  auto m = parent.def_submodule(name, doc);
  m.attr("__package__") = parent.attr("__name__");
  return m;
}

PYBIND11_MODULE(cr, m) {
  m.attr("__name__") = "cr";
  m.attr("__package__") = m.attr("__name__");
  m.doc() = "pybind11 and cmake example";
  m.def("add", &add, "A function which adds two numbers");
  m.def("sub", [](double i, double j) { return i - j; }, "A function which subtract two numbers");
  m.def("add_list", &add_vector);
  auto custom_class = submodule(m, "custom_class", "custom class");
  auto numpy_functions = submodule(m, "numpy_functions", "numpy functions");
  init_custom_class(custom_class);
  init_numpy_functions(numpy_functions);
  m.def("get_str", []() {
    std::string x = "你好";
    printUtf8Bytes(x);
    return x;
  });
  m.def("get_bytes", []() { return pybind11::bytes("你好"); });
}
