#include <pybind11/pybind11.h>

namespace py = pybind11;

PyObject *python_function;

void register_function(py::object function) { python_function = function.ptr(); }
void call_function() { PyObject_CallObject(python_function, nullptr); }

PYBIND11_MODULE(hello_world, m) {
  m.attr("__name__") = "hello_world";
  m.attr("__package__") = m.attr("__name__");
  m.def("register_function", &register_function);
  m.def("call_function", &call_function);
}