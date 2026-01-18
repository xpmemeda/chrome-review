#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <cuda_runtime.h>
#include <unordered_map>

#include "./module.h"
#include "helper.h"

namespace comm {

#define CUDA_CHECK(expr)                                              \
  do {                                                                \
    cudaError_t err = (expr);                                         \
    if (err != cudaSuccess) {                                         \
      TORCH_CHECK(false, #expr " failed: ", cudaGetErrorString(err)); \
    }                                                                 \
  } while (0)

CudaMappedTensorManager::~CudaMappedTensorManager() {
  for (auto [hptr, _] : h2ds_) {
    cudaHostUnregister(hptr);
  }
}
CudaMappedTensorManager* CudaMappedTensorManager::get() {
  static CudaMappedTensorManager manager;
  return &manager;
}

void CudaMappedTensorManager::_register(at::Tensor& x) {
  void* hraw = x.data_ptr();
  auto it = h2ds_.find(hraw);
  if (it != h2ds_.end()) {
    printf("%p has regitered.", hraw);
    return;
  }

  CUDA_CHECK(cudaHostRegister(hraw, x.nbytes(), cudaHostRegisterMapped));
  void* draw = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&draw, hraw, 0));
  h2ds_[hraw] = draw;
  printf("CudaMappedTensorManager registe ptr %p\n", x.data_ptr());
}

void CudaMappedTensorManager::_release(at::Tensor& x) {
  auto it = h2ds_.find(x.data_ptr());
  if (it == h2ds_.end()) {
    printf("%p not found.", x.data_ptr());
    return;
  }
  CUDA_CHECK(cudaHostUnregister(x.data_ptr()));
  h2ds_.erase(x.data_ptr());
  printf("CudaMappedTensorManager release %p\n", x.data_ptr());
}

void* CudaMappedTensorManager::get(void* h_ptr) {
  auto it = h2ds_.find(h_ptr);
  if (it == h2ds_.end()) {
    printf("%p not regitered before.");
    throw std::runtime_error("");
  }

  return it->second;
}

static cr::Register _([](pybind11::module& m) {
  pybind11::class_<CudaMappedTensorManager>(m, "CudaMappedTensorManager")
      .def_static(
          "get", []() { return CudaMappedTensorManager::get(); }, py::return_value_policy::reference)
      .def("register", &CudaMappedTensorManager::_register)
      .def("release", &CudaMappedTensorManager::_release);
});

}  // namespace comm
