#include <stdio.h>
#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

struct Object {
  Object() { printf("in ctor, this: %p\n", this); }
  void showThis() { printf("this: %p\n", this); }
};

class SharedObject : public Object {};
class UniqueObject : public Object {};

PYBIND11_MODULE(smartptr, m) {
  m.attr("__name__") = "smartptr";
  m.attr("__package__") = "smartptr";

  pybind11::class_<SharedObject, std::shared_ptr<SharedObject>>(m, "SharedObject")
      .def(pybind11::init<>())
      .def("show_this", &SharedObject::showThis)
      .def_static("accept_object", [](SharedObject x) { printf("accept object: this = %p\n", &x); })
      .def_static("accept_ref", [](SharedObject& x) { printf("accept ref: this = %p\n", &x); })
      .def_static("accept_raw_ptr", [](SharedObject* x) { printf("accept shared raw ptr: this = %p\n", x); })
      .def_static("accept_shared_ptr",
          [](std::shared_ptr<SharedObject> x) { printf("accept shared raw ptr: this = %p\n", x.get()); });

  pybind11::class_<UniqueObject, std::unique_ptr<UniqueObject>>(m, "UniqueObject")
      .def(pybind11::init<>())
      .def("show_this", &UniqueObject::showThis)
      .def_static("accept_object", [](UniqueObject x) { printf("accept object: this = %p\n", &x); })
      .def_static("accept_ref", [](UniqueObject& x) { printf("accept ref: this = %p\n", &x); })
      .def_static("accept_raw_ptr", [](UniqueObject* x) { printf("accept raw ptr: this = %p\n", x); })
      // .def_static("accept_unique_ptr",     // cannot compile.
      //     [](std::unique_ptr<UniqueObject> x) { printf("accept unique ptr: this = %p\n", x.get()); })
      ;
}