#include <iostream>
#include <thread>

#include "cuda.h"
#include "cuda_runtime.h"

#include "exceptions.h"

void print_device_and_context() {
  int device = -1;
  cr::check_cuda_err(cudaGetDevice(&device), "cudaGetDevice");
  std::cout << "\tdevice = " << device << std::endl;
  CUcontext c;
  cr::check_cuda_err(cuCtxGetCurrent(&c), "cuCtxGetCurrent");
  std::cout << "\tcontext = " << c << std::endl;
}

void fn1() {
  CUcontext c;
  try {
    cr::check_cuda_err(cuCtxGetCurrent(&c), "cuCtxGetCurrent");
  } catch (...) {
    std::cout << "fn1: 在使用cuda driver api之前必须调用cuInit" << std::endl;
  }
}

void fn2() {
  cr::check_cuda_err(cuInit(0), "cuInit");
  std::cout << "fn2: 新线程默认使用设备0，不绑定context" << std::endl;
  print_device_and_context();
}

void fn3() {
  cr::check_cuda_err(cudaSetDevice(1), "cudaSetDevice");
  std::cout << "fn3: cudaSetDevice为线程绑定设备，并且把context绑定为设备主context" << std::endl;
  print_device_and_context();
  std::thread([]() {
    cr::check_cuda_err(cudaSetDevice(1), "cudaSetDevice");
    std::cout << "fn3: 即使是不同的线程，在使用cudaSetDevice绑定同一个线程后，也会共享同一个设备主context" << std::endl;
    print_device_and_context();
  }).join();
}

void fn4() {
  cr::check_cuda_err(cudaFree(nullptr), "cudaFree");
  std::cout << "fn4: 随便调用一个cuda runtime api会绑定当前线程到设备0的主context上面" << std::endl;
  print_device_and_context();
}

void fn5() {
  cr::check_cuda_err(cudaSetDevice(1), "cudaSetDevice");
  CUcontext primary_context;
  cr::check_cuda_err(cuDevicePrimaryCtxRetain(&primary_context, 0));
  cr::check_cuda_err(cuCtxSetCurrent(primary_context));
  std::cout << "fn5: "
               "绑定设备的本质上是Context，即使先用cudaSetDevice指定了设备1，但是context切换到设备0之后，绑定的设备也切"
               "换到设备0"
            << std::endl;
  print_device_and_context();
}

int main() {
  std::thread(fn1).join();
  std::thread(fn2).join();
  std::thread(fn3).join();
  std::thread(fn4).join();
  std::thread(fn5).join();
  return 0;
}
