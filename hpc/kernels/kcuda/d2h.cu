#include <ATen/Tensor.h>
#include <cuda_runtime.h>

#include "./module.h"
#include "helper.h"

namespace {

#define CUDA_CHECK(expr)                                              \
  do {                                                                \
    cudaError_t err = (expr);                                         \
    if (err != cudaSuccess) {                                         \
      TORCH_CHECK(false, #expr " failed: ", cudaGetErrorString(err)); \
    }                                                                 \
  } while (0)

/**
 * src [m, n]
 * dst [m, n]
 */
template <typename T, typename vec_t, unsigned FEATURES, unsigned THREADS>
__global__ void d2h_kernel(T* src, T* dst, int m, int ldm) {
  constexpr int NUMEL = sizeof(vec_t) / sizeof(T);
  constexpr int NUMLD = FEATURES / NUMEL;
  static_assert(FEATURES % NUMEL == 0);

  auto vec_src = reinterpret_cast<vec_t*>(src + blockIdx.x * ldm);
  auto vec_dst = reinterpret_cast<vec_t*>(dst + blockIdx.x * ldm);

#pragma unroll
  for (int i = threadIdx.x; i < NUMLD; i += blockDim.x) {
    vec_dst[i] = vec_src[i];
  }
}

void d2h(at::Tensor& src, at::Tensor& dst) {
  unsigned m = src.size(0);
  unsigned n = src.size(1);
  unsigned ldm = src.stride(0);

  using vec_t = uint4;

#define DISPATCH_BY_FEATURES(T, FEATURES)                                                   \
  case FEATURES: {                                                                          \
    constexpr unsigned VECS = FEATURES / (sizeof(vec_t) / sizeof(T));                       \
    constexpr unsigned THREADS = ((VECS >= 1024 ? 1024 : VECS) + 31) / 32 * 32;             \
    const unsigned blocks = m;                                                              \
    const unsigned threads = THREADS;                                                       \
    d2h_kernel<T, vec_t, FEATURES, THREADS><<<blocks, threads>>>(src_ptr, dst_ptr, m, ldm); \
    CUDA_CHECK(cudaGetLastError());                                                         \
    break;                                                                                  \
  }

#define DISPATCH_BY_ELEMENT_SIZE(ELEM_SIZE, T)                                                    \
  case ELEM_SIZE: {                                                                               \
    T* src_ptr = src.data_ptr<T>();                                                               \
    T* dst_ptr = reinterpret_cast<T*>(comm::CudaMappedTensorManager::get()->get(dst.data_ptr())); \
    switch (n) {                                                                                  \
      DISPATCH_BY_FEATURES(T, 1024);                                                              \
      DISPATCH_BY_FEATURES(T, 2048);                                                              \
      DISPATCH_BY_FEATURES(T, 4096);                                                              \
      default:                                                                                    \
        throw std::runtime_error("unsupported n " + std::to_string(n));                           \
    }                                                                                             \
    break;                                                                                        \
  }

  switch (src.element_size()) {
    DISPATCH_BY_ELEMENT_SIZE(1, uint8_t);
    DISPATCH_BY_ELEMENT_SIZE(2, uint16_t);
    DISPATCH_BY_ELEMENT_SIZE(4, uint32_t);
    default:
      throw std::runtime_error("unsupported element size " + std::to_string(src.element_size()));
  }
}

static cr::Register _([](pybind11::module& m) { m.def("d2h", &d2h); });

}  // namespace
