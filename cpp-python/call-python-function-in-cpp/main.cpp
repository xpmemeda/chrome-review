#include <iostream>
#include <string>
#include "pybind11/embed.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

PYBIND11_MODULE(hello_world, m) {
  m.attr("__name__") = "hello_world";
  m.attr("__package__") = m.attr("__name__");

  struct A {
    int x;
    std::string y;
  };
  pybind11::class_<A>(m, "A").def(pybind11::init<>()).def(pybind11::init<const A&>());

  m.def("call_python_function_with_args_and_kwargs", [](pybind11::object function) {
    using namespace pybind11::literals;
    auto builtins = pybind11::module_::import("builtins");
    auto print = builtins.attr("print");
    auto r = function(true, pybind11::none(), pybind11::bytes("1"), "x"_a = 1, "y"_a = "1", "z"_a = A({1, "1"}));
    print(r);
  });

  m.def("list", [](pybind11::object function) {
    auto builtins = pybind11::module_::import("builtins");
    auto print = builtins.attr("print");
    print(function);
    {
      pybind11::list x;
      x.append(pybind11::int_(0));
      x.append(1);  // auto cast 1 -> pybind11::int_
      auto r = function(x);
      print("using pybind11 api"), print(r);
      std::cout << "[";
      for (size_t i = 0; i < r.cast<pybind11::list>().size(); ++i) {
        std::cout << r.cast<pybind11::list>()[i].cast<int>() << ",]"[i + 1 == r.cast<pybind11::list>().size()];
        std::cout << " \n"[i + 1 == r.cast<pybind11::list>().size()];
      }
    }
    {
      auto x = PyList_New(0);
      PyList_Append(x, PyLong_FromLong(0));
      PyList_Append(x, PyLong_FromLong(1));
      auto r = PyObject_CallObject(function.ptr(), PyTuple_Pack(1, x));
      print("using cpython raw api");
      print(pybind11::reinterpret_borrow<pybind11::object>(r));
      std::cout << "[";
      for (size_t i = 0; i < PyList_Size(r); ++i) {
        int v = i + 1 == PyList_Size(r);
        std::cout << PyLong_AsLong(PyList_GetItem(r, i)) << ",]"[v] << " \n"[v];
      }
    }
  });

  m.def("dict", [](pybind11::object function) {
    auto builtins = pybind11::module_::import("builtins");
    auto print = builtins.attr("print");
    print(function);
    {
      pybind11::dict x;
      x[pybind11::str("a", 1)] = pybind11::int_(1);
      x[pybind11::str("b", 1)] = pybind11::int_(2);
      auto r = function(x);
      print("using pybind11 api");
      print(r);
      for (auto item : r.cast<pybind11::dict>()) {
        auto key = item.first.cast<std::string>();
        auto val = item.second.cast<int>();
        std::cout << "key: " << key << ", val:" << val << std::endl;
      }
    }
    {
      auto x = PyDict_New();
      PyDict_SetItem(x, PyUnicode_FromString("a"), PyLong_FromLong(0));
      PyDict_SetItem(x, PyUnicode_FromString("b"), PyLong_FromLong(1));
      auto r = PyObject_CallObject(function.ptr(), PyTuple_Pack(1, x));
      print("using cpython raw api");
      print(pybind11::reinterpret_borrow<pybind11::object>(r));
      std::cout << "{";
      auto keys = PyDict_Keys(r);
      for (size_t i = 0; i < PyList_Size(keys); ++i) {
        int index = i + 1 == PyList_Size(keys);
        auto k = PyList_GetItem(keys, i);
        auto v = PyDict_GetItem(r, k);
        std::cout << PyUnicode_AS_DATA(k) << ": " << PyLong_AsLong(v) << ",}"[index] << " \n"[index];
      }
    }
  });

  m.def("custom", [&m](pybind11::object function) {
    auto builtins = pybind11::module_::import("builtins");
    auto print = builtins.attr("print");
    print(function);
    {
      auto r = function(A({1, std::string({"1"})}));
      print("using pybind11 api");
      print(r);
      std::cout << "A.x=" << r.cast<A>().x << ", A.y=" << r.cast<A>().y << std::endl;
    }
    {
      auto x = pybind11::module::import("hello_world").attr("A")(A({0, "1"}));
      auto r = PyObject_CallObject(function.ptr(), PyTuple_Pack(1, x.ptr()));
      print("using cpython raw api");
      print(pybind11::reinterpret_borrow<pybind11::object>(r));
      std::cout << "A.x=" << pybind11::reinterpret_borrow<pybind11::object>(r).cast<A>().x
                << ", A.y=" << pybind11::reinterpret_borrow<pybind11::object>(r).cast<A>().y << std::endl;
    }
  });
}