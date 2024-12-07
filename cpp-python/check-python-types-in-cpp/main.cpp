#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void print(PyObject *obj) {
  Py_INCREF(obj);

  if (PyLong_Check(obj)) {
    long v = PyLong_AsLong(obj);
    std::cout << "long: v = " << v << std::endl;
  }

  if (PyFloat_Check(obj)) {
    double v = PyFloat_AsDouble(obj);
    std::cout << "double: v = " << v << std::endl;
  }

  if (PyBytes_Check(obj)) {
    std::string v = PyBytes_AsString(obj);
    std::cout << "bytes: v = " << v << std::endl;
  }

  if (PyUnicode_Check(obj)) {
    std::string v = (const char *)PyUnicode_DATA(obj);
    std::cout << "unicode: v = " << v << std::endl;
  }

  if (PyList_Check(obj)) {
    std::cout << "list: [" << std::endl;
    for (Py_ssize_t i = 0; i < PyList_Size(obj); ++i) {
      std::cout << "  ";
      print(PyList_GetItem(obj, i));
    }
    std::cout << "]" << std::endl;
  }

  Py_DECREF(obj);
}

PYBIND11_MODULE(hello_world, m) {
  m.attr("__name__") = "hello_world";
  m.attr("__package__") = m.attr("__name__");
  m.def("print", [](py::object obj) {
    PyObject *raw_obj = obj.ptr();
    print(raw_obj);
  });
}