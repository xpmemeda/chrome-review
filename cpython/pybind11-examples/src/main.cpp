#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <assert.h>

#include "numpy_functions.h"
#include "custom_class.h"


double add(double i, double j) {
    return i + j;
}


std::vector<double> add_vector(const std::vector<double>& x, const std::vector<double>& y) {
    assert(x.size() == y.size());
    std::vector<double> z(x.size());
    for (int i = 0; i < x.size(); ++i) {
        z[i] = x[i] + y[i];
    }
    return z;
}


namespace py=pybind11;


py::module submodule(py::module parent, const char* name, const char* doc) {
    auto m = parent.def_submodule(name, doc);
    m.attr("__package__") = parent.attr("__name__");
    return m;
}


PYBIND11_MODULE(example, m) {
    m.attr("__name__") = "example";
    m.attr("__package__") = m.attr("__name__");
    m.doc() = "pybind11 and cmake example";
    m.def("add", &add, "A function which adds two numbers");
    m.def("sub", [](double i, double j) { return i - j; }, "A function which subtract two numbers");
    m.def("add_list", &add_vector);
    auto custom_class = submodule(m, "custom_class", "custom class");
    auto numpy_functions = submodule(m, "numpy_functions", "numpy functions");
    init_custom_class(custom_class);
    init_numpy_functions(numpy_functions);
}
