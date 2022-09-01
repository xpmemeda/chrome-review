#include <iostream>
#include <vector>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "numpy_functions.h"


std::vector<double> multiply(const std::vector<double>& left, const std::vector<double>& right) {
    if (left.size() != right.size()) {
        throw std::runtime_error("the length of inputs shoulb be equal");
    }
    std::vector<double> ret(left.size());
    for (size_t i = 0; i < ret.size(); ++i) {
        ret[i] = left[i] * right[i];
    }
    return ret;
}

std::vector<int> reduce(const std::vector<int>& input, int stride) {
    std::vector<int> output(input.size() / stride, 0);
    for (int i = 0; i < output.size(); ++i) {
        for (int j = 0; j < stride; ++j) {
            output[i] = output[i] + input[i * stride + j];
        }
    }
    return output;
}


namespace py = pybind11;

// wrap C++ function with NumPy array IO
using np_double =py::array_t<double, py::array::c_style | py::array::forcecast>;
py::array_t<double> py_multiply(np_double left, np_double right) {
    if (left.ndim() != 1 && right.ndim() != 1) {
        throw std::runtime_error("Inputs should be 1-D NumPy array");
    }
    // allocate std::vector (to pass to the C++ function)
    std::vector<double> vec_left(left.size());
    std::vector<double> vec_right(right.size());
    // py::array -> std::vector
    std::memcpy(vec_left.data(), left.data(), left.size() * sizeof(double));
    std::memcpy(vec_right.data(), right.data(), right.size() * sizeof(double));
    // call C++ function
    std::vector<double> result_vec = multiply(vec_left, vec_right);
    // allocate py::array (to pass the result of C++ function to Python)
    auto result  = py::array_t<double>(left.size());
    auto result_buffer = result.request();
    double* result_ptr = (double*)result_buffer.ptr;
    // std::vector -> py::array
    std::memcpy(result_ptr, result_vec.data(), result_vec.size() * sizeof(double));

    return result;
}


py::array py_reduce(py::array_t<int> array) {
    if (array.ndim() != 2) {
        throw std::runtime_error("Input should be 2-D NumPy array");
    }
    std::vector<int> input(array.size());
    int stride = array.shape()[1];
    std::memcpy(input.data(), array.data(), array.size() * sizeof(int));
    std::vector<int> output = reduce(input, stride);
    ssize_t ndim = 1;
    std::vector<ssize_t> shape = {array.shape()[0]};
    std::vector<ssize_t> strides = {sizeof(int)};
    return py::array(py::buffer_info(
        output.data(),
        sizeof(int),
        py::format_descriptor<int>::format(),
        ndim,
        shape,
        strides
    ));
}

void init_numpy_functions(py::module m) {
    m.def("multiply", &py_multiply);
    m.def("reduce", &py_reduce);
}
