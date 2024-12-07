#include <string>
#include <pybind11/pybind11.h>
#include "custom_class.h"


Pet::Pet(const std::string& name) : _name(name), public_name(name) {}
void Pet::setName(const std::string& name) { _name = name; }
const std::string& Pet::getName() const { return _name; }

Dog::Dog(const std::string& name) : Pet(name) {}
std::string Dog::bark() { return "woof"; }

namespace py = pybind11;

void init_custom_class(py::module m) {
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string&>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName)
        .def_property("name", &Pet::getName, &Pet::setName)
        .def_readwrite("public_name", &Pet::public_name)
        .def("__repr__",
            [](const Pet& a) {
                return "<example.Pet public named '" + a.public_name + "'>";
            }
        );
    py::class_<Dog, Pet>(m, "Dot")
        .def(py::init<const std::string&>())
        .def("bark", &Dog::bark);
    m.def("get_dog", [](std::string name) { return std::unique_ptr<Dog>(new Dog(name)); });
}
