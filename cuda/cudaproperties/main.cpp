#include <iostream>
#include <tuple>
#include "cuda_runtime.h"

#include "exceptions.h"

void print_hardware_properties() {
  int device_count;
  cr::check_cuda_err(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    cr::check_cuda_err(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties");
    std::cout << "device " << i << ": " << prop.name << " Arch " << prop.major << '.' << prop.minor << " SM "
              << prop.multiProcessorCount << " Blocks/SM " << prop.maxBlocksPerMultiProcessor << " Warps/SM "
              << prop.maxThreadsPerMultiProcessor / 32 << " Regs/SM " << prop.regsPerMultiprocessor << " Smem/SM "
              << prop.sharedMemPerMultiprocessor / 1024 << "KB" << std::endl;
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
    cr::check_cuda_err(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    auto [major, minor] = convert_int_to_cuda_version(driver_version);
    std::cout << "driver: " << major << '.' << minor << std::endl;
  }

  {
    int runtime_version;
    cr::check_cuda_err(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    auto [major, minor] = convert_int_to_cuda_version(runtime_version);
    std::cout << "runtime: " << major << '.' << minor << std::endl;
  }
}

int main() {
  print_hardware_properties();
  print_software_properties();
  return 0;
}