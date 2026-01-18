#include <ATen/Tensor.h>
#include <cuda_runtime.h>

#include "./module.h"

namespace {

__global__ void process_image(int* image, int height, int width) {
  int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_idx < height * width) {
    image[thread_idx] = image[thread_idx] * 2;
  }
}

void launcher(int* ptr, int height, int width) {
  unsigned block_size = 128;
  unsigned grid_size = (height * width + block_size - 1) / block_size;
  process_image<<<grid_size, block_size>>>(ptr, height, width);
}

void zero_copy_y(at::Tensor& image) {
  class CudaRegisterGuard {
   public:
    ~CudaRegisterGuard() {
      for (auto [hptr, dptr] : h2ds_) {
        cudaHostUnregister(hptr);
      }
    }

    int* get(at::Tensor& x) {
      auto hptr = static_cast<int*>(x.data_ptr());
      auto it = h2ds_.find(hptr);
      if (it == h2ds_.end()) {
        cudaHostRegister(hptr, x.nbytes(), cudaHostRegisterMapped);
        cudaHostGetDevicePointer(&h2ds_[hptr], hptr, 0);
      }
      return h2ds_[hptr];
    }

   private:
    std::unordered_map<int*, int*> h2ds_;
  };

  static CudaRegisterGuard bridge;
  auto dptr = bridge.get(image);

  launcher(dptr, image.size(0), image.size(1));
  cudaDeviceSynchronize();
}

void zero_copy_n(at::Tensor& image) {
  class Bridge {
   public:
    Bridge() { printf("Bridge Ctor.\n"); }
    ~Bridge() { printf("Bridge Dtor.\n"); }

    int* get(at::Tensor& ht) {
      auto it = h2ds_.find(ht.data_ptr());
      if (it == h2ds_.end()) {
        h2ds_[ht.data_ptr()] = ht.to(at::kCUDA);
      }
      return static_cast<int*>(h2ds_[ht.data_ptr()].data_ptr());
    }

    void h2d(at::Tensor& ht) {
      auto& dt = h2ds_[ht.data_ptr()];
      dt.copy_(ht);
    }
    void d2h(at::Tensor& ht) {
      auto& dt = h2ds_[ht.data_ptr()];
      ht.copy_(dt);
    }

   private:
    std::unordered_map<void*, at::Tensor> h2ds_;
  };

  static Bridge bridge;
  auto dptr = bridge.get(image);

  bridge.h2d(image);
  launcher(dptr, image.size(0), image.size(1));
  bridge.d2h(image);
}

static cr::Register _([](pybind11::module& m) {
  m.def("zero_copy_y", &zero_copy_y);
  m.def("zero_copy_n", &zero_copy_n);
});

}  // namespace
