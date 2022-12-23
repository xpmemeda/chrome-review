#include <iostream>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

int init_numpy() {
  import_array();
  return 0;
}

const static int numpy_initialized = init_numpy();

void print(PyObject *obj) {
  Py_INCREF(obj);

  if (PyLong_Check(obj)) {
    long v = PyLong_AsLong(obj);
    std::cout << "long: v = " << v << std::endl;
  }

  if (PyUnicode_Check(obj)) {
    std::string v = (const char *)PyUnicode_DATA(obj);
    std::cout << "unicode: v = " << v << std::endl;
  }

  if (PyArray_Check(obj)) {
    void *array_data = PyArray_DATA(obj);
    auto item_size = PyArray_ITEMSIZE(obj);
    std::cout << "item_size: " << item_size << std::endl;
    for (npy_intp i = 0; i < PyArray_Size(obj); ++i) {
      print(PyArray_GETITEM(obj, array_data + item_size * i));
    }
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