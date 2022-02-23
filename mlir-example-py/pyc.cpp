#include <pybind11/pybind11.h>

namespace py=pybind11;

int func_main();

PYBIND11_MODULE(mlir_example_py, m) {
    m.def("func_main", &func_main, "func_main");
}
