#include <iostream>
#include <tuple>
#include "cuda_runtime.h"

class CUDARuntimeError : public std::exception {
  cudaError_t _status;

 public:
  CUDARuntimeError(cudaError_t status) : _status(status) {}

  const char* what() const noexcept override { return cudaGetErrorString(_status); }
};

void print_hardware_properties() {
  int device_count;
  {
    cudaError_t ret = cudaGetDeviceCount(&device_count);
    if (ret != cudaSuccess) {
      throw CUDARuntimeError(ret);
    }
  }

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    cudaError_t ret = cudaGetDeviceProperties(&prop, i);
    if (ret != cudaSuccess) {
      throw CUDARuntimeError(ret);
    }
    std::cout << "device " << i << ": arch " << prop.major << '.' << prop.minor << std::endl;
  }
}

void print_software_properties() {
  auto convert_int_to_cuda_version = [&](int num) {
    int major = num / 1000;
    int minor = (num % 100) / 10;
    return std::make_tuple(major, minor);
  };

  {
    int driver_version;
    cudaError_t ret = cudaDriverGetVersion(&driver_version);
    if (ret != cudaSuccess) {
      throw CUDARuntimeError(ret);
    }
    auto [major, minor] = convert_int_to_cuda_version(driver_version);
    std::cout << "driver: " << major << '.' << minor << std::endl;
  }

  {
    int runtime_version;
    cudaError_t ret = cudaRuntimeGetVersion(&runtime_version);
    if (ret != cudaSuccess) {
      throw CUDARuntimeError(ret);
    }
    auto [major, minor] = convert_int_to_cuda_version(runtime_version);
    std::cout << "runtime: " << major << '.' << minor << std::endl;
  }
}

int main() {
  print_hardware_properties();
  print_software_properties();
  return 0;
}