#include <pybind11/pybind11.h>

struct Dog {
    Dog(const std::string& name, int age) : name(name), age(age) {}
    std::string name;
    int age;
};

PYBIND11_MODULE(example, m) {
    pybind11::class_<Dog>(m, "Dog")
        .def(pybind11::init<const std::string&, int>())
        .def_readwrite("name", &Dog::name)
        .def_readwrite("age", &Dog::age)
        .def("__call__", [](const Dog &) { return "woof...woof..."; })
        ;
}
